//! Static module graph discovery and module-specifier resolution for WrenJIT.
//!
//! Resolution happens before WIR/bytecode generation. The VM therefore pays no
//! filesystem/package-manager cost when executing ImportModule: imports are
//! lowered to canonical StringId values and resolved by the VM's dense module map.
//!
//! Supported specifiers:
//! - `./foo`, `../foo.wren`, `foo`  -> importer-relative source modules.
//! - `@lib/pkg[/subpath]`           -> package roots.
//! - `@std/name[/subpath]`          -> standard-library roots.
//!
//! For every filesystem base we try the exact path, `<path>.wren`, and
//! `<path>/index.wren`. Package/std specifiers may not contain `..` and symlink
//! escapes outside their configured root are rejected.

use crate::wren_front::{self, Diagnostic, ModuleAst};
use crate::wren_ir::Program;
use crate::wren_lowerer::{self, LowerOptions, LoweredModule};

use hashbrown::{HashMap, HashSet};
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

const SOURCE_EXTENSION: &str = "wren";
const PROJECT_MARKERS: &[&str] = &["wrenjit.toml", "wren.toml", "Wren.toml", ".wrenjit", ".git"];

#[derive(Clone, Debug)]
pub struct ModuleResolverConfig {
    pub project_root: PathBuf,
    pub package_roots: Vec<PathBuf>,
    pub std_roots: Vec<PathBuf>,
}

impl ModuleResolverConfig {
    pub fn for_entry(entry: &Path) -> Result<Self, ModuleError> {
        let entry = canonical_file(entry)?;
        let start = entry.parent().unwrap_or_else(|| Path::new("."));
        let project_root = discover_project_root(start);

        let mut package_roots = Vec::new();
        package_roots.push(project_root.join(".wrenjit").join("packages"));
        extend_env_paths(&mut package_roots, "WRENJIT_LIB_PATH");

        let mut std_roots = Vec::new();
        std_roots.push(project_root.join(".wrenjit").join("std"));
        extend_env_paths(&mut std_roots, "WRENJIT_STD_PATH");

        if let Some(home) = env::var_os("WRENJIT_HOME") {
            let home = PathBuf::from(home);
            package_roots.push(home.join("packages"));
            std_roots.push(home.join("std"));
        }

        if let Ok(executable) = env::current_exe() {
            if let Some(bin) = executable.parent() {
                std_roots.push(bin.join("std"));
                std_roots.push(bin.join("share").join("wrenjit").join("std"));
            }
        }

        dedupe_roots(&mut package_roots);
        dedupe_roots(&mut std_roots);

        Ok(Self {
            project_root,
            package_roots,
            std_roots,
        })
    }

    pub fn with_package_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.package_roots.insert(0, root.into());
        dedupe_roots(&mut self.package_roots);
        self
    }

    pub fn with_std_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.std_roots.insert(0, root.into());
        dedupe_roots(&mut self.std_roots);
        self
    }
}

#[derive(Clone, Debug)]
pub struct ModuleResolver {
    config: ModuleResolverConfig,
}

#[derive(Clone, Debug)]
pub struct ResolvedModule {
    pub path: PathBuf,
    pub canonical_name: String,
}

impl ModuleResolver {
    pub fn new(config: ModuleResolverConfig) -> Self {
        Self { config }
    }

    pub fn for_entry(entry: &Path) -> Result<Self, ModuleError> {
        Ok(Self::new(ModuleResolverConfig::for_entry(entry)?))
    }

    pub fn config(&self) -> &ModuleResolverConfig {
        &self.config
    }

    pub fn resolve_entry(&self, entry: &Path) -> Result<ResolvedModule, ModuleError> {
        let path = canonical_file(entry)?;
        Ok(ResolvedModule {
            canonical_name: self.identity_for_path(&path),
            path,
        })
    }

    pub fn resolve_import(
        &self,
        importer: &Path,
        specifier: &str,
    ) -> Result<ResolvedModule, ModuleError> {
        if specifier.is_empty() {
            return Err(ModuleError::InvalidSpecifier {
                importer: importer.to_owned(),
                specifier: specifier.to_owned(),
                reason: "module specifier cannot be empty".to_owned(),
            });
        }
        if specifier.as_bytes().contains(&0) {
            return Err(ModuleError::InvalidSpecifier {
                importer: importer.to_owned(),
                specifier: specifier.to_owned(),
                reason: "module specifier contains a NUL byte".to_owned(),
            });
        }

        let mut searched = Vec::new();
        let path = if let Some(tail) = specifier.strip_prefix("@lib/") {
            self.resolve_in_roots(importer, specifier, tail, &self.config.package_roots, &mut searched)?
        } else if specifier == "@lib" {
            return Err(ModuleError::InvalidSpecifier {
                importer: importer.to_owned(),
                specifier: specifier.to_owned(),
                reason: "`@lib` must be followed by a package name".to_owned(),
            });
        } else if let Some(tail) = specifier.strip_prefix("@std/") {
            self.resolve_in_roots(importer, specifier, tail, &self.config.std_roots, &mut searched)?
        } else if specifier == "@std" {
            return Err(ModuleError::InvalidSpecifier {
                importer: importer.to_owned(),
                specifier: specifier.to_owned(),
                reason: "`@std` must be followed by a module name".to_owned(),
            });
        } else if specifier.starts_with('@') {
            return Err(ModuleError::InvalidSpecifier {
                importer: importer.to_owned(),
                specifier: specifier.to_owned(),
                reason: "unknown module namespace (expected @lib/ or @std/)".to_owned(),
            });
        } else {
            let requested = Path::new(specifier);
            if requested.is_absolute() {
                return Err(ModuleError::InvalidSpecifier {
                    importer: importer.to_owned(),
                    specifier: specifier.to_owned(),
                    reason: "absolute import paths are not portable; use a relative, @lib/, or @std/ specifier".to_owned(),
                });
            }
            let base = importer
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(requested);
            resolve_candidate_set(&base, &mut searched).ok_or_else(|| ModuleError::Resolve {
                importer: importer.to_owned(),
                specifier: specifier.to_owned(),
                searched: searched.clone(),
            })?
        };

        Ok(ResolvedModule {
            canonical_name: self.identity_for_path(&path),
            path,
        })
    }

    fn resolve_in_roots(
        &self,
        importer: &Path,
        specifier: &str,
        tail: &str,
        roots: &[PathBuf],
        searched: &mut Vec<PathBuf>,
    ) -> Result<PathBuf, ModuleError> {
        validate_namespaced_tail(importer, specifier, tail)?;

        for configured_root in roots {
            let root = match canonical_directory(configured_root) {
                Some(root) => root,
                None => {
                    // Keep useful diagnostics even when the package manager has
                    // not created this root yet.
                    searched.extend(candidate_paths(&configured_root.join(tail)));
                    continue;
                }
            };
            let base = root.join(tail);
            if let Some(path) = resolve_candidate_set(&base, searched) {
                // Canonicalization resolves symlinks. Namespace imports must not
                // use a symlink to escape the package/std root.
                if path.starts_with(&root) {
                    return Ok(path);
                }
                return Err(ModuleError::InvalidSpecifier {
                    importer: importer.to_owned(),
                    specifier: specifier.to_owned(),
                    reason: format!(
                        "resolved path '{}' escapes namespace root '{}'",
                        path.display(),
                        root.display()
                    ),
                });
            }
        }

        Err(ModuleError::Resolve {
            importer: importer.to_owned(),
            specifier: specifier.to_owned(),
            searched: searched.clone(),
        })
    }

    fn identity_for_path(&self, path: &Path) -> String {
        // Check namespace roots before project_root because local package/std
        // directories commonly live inside the project tree.
        for root in &self.config.std_roots {
            if let Some(root) = canonical_directory(root) {
                if let Ok(relative) = path.strip_prefix(&root) {
                    return format!("@std/{}", slash_path(relative));
                }
            }
        }
        for root in &self.config.package_roots {
            if let Some(root) = canonical_directory(root) {
                if let Ok(relative) = path.strip_prefix(&root) {
                    return format!("@lib/{}", slash_path(relative));
                }
            }
        }

        let project_root = canonical_directory(&self.config.project_root)
            .unwrap_or_else(|| self.config.project_root.clone());
        if let Ok(relative) = path.strip_prefix(&project_root) {
            return format!("file:{}", slash_path(relative));
        }

        format!("file://{}", slash_path(path))
    }
}

#[derive(Debug)]
struct ParsedModule {
    path: PathBuf,
    canonical_name: String,
    source: String,
    ast: ModuleAst,
    import_map: HashMap<String, String>,
}

#[derive(Debug)]
pub struct ModuleDiagnosticSource {
    pub source_name: String,
    pub source: String,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug)]
pub struct CompiledModuleGraph {
    pub entry: LoweredModule,
    pub entry_name: String,
    pub module_count: usize,
    diagnostics: Vec<ModuleDiagnosticSource>,
}

impl CompiledModuleGraph {
    pub fn diagnostics(&self) -> &[ModuleDiagnosticSource] {
        &self.diagnostics
    }

    pub fn emit_diagnostics(&self) -> io::Result<()> {
        emit_diagnostic_sources(&self.diagnostics)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModuleErrorKind {
    Io,
    Compile,
}

#[derive(Debug)]
pub enum ModuleError {
    Io {
        path: PathBuf,
        error: io::Error,
    },
    Resolve {
        importer: PathBuf,
        specifier: String,
        searched: Vec<PathBuf>,
    },
    InvalidSpecifier {
        importer: PathBuf,
        specifier: String,
        reason: String,
    },
    Diagnostics {
        sources: Vec<ModuleDiagnosticSource>,
    },
}

impl ModuleError {
    pub fn kind(&self) -> ModuleErrorKind {
        match self {
            ModuleError::Io { .. } => ModuleErrorKind::Io,
            ModuleError::Resolve { .. }
            | ModuleError::InvalidSpecifier { .. }
            | ModuleError::Diagnostics { .. } => ModuleErrorKind::Compile,
        }
    }

    pub fn emit(&self) -> io::Result<()> {
        match self {
            ModuleError::Diagnostics { sources } => emit_diagnostic_sources(sources),
            _ => {
                eprintln!("{self}");
                Ok(())
            }
        }
    }
}

impl fmt::Display for ModuleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModuleError::Io { path, error } => {
                write!(formatter, "failed to read '{}': {error}", path.display())
            }
            ModuleError::Resolve {
                importer,
                specifier,
                searched,
            } => {
                write!(
                    formatter,
                    "cannot resolve module '{specifier}' imported from '{}'",
                    importer.display()
                )?;
                if !searched.is_empty() {
                    write!(formatter, "\nsearched:")?;
                    for path in searched {
                        write!(formatter, "\n  - {}", path.display())?;
                    }
                }
                Ok(())
            }
            ModuleError::InvalidSpecifier {
                importer,
                specifier,
                reason,
            } => write!(
                formatter,
                "invalid module specifier '{specifier}' in '{}': {reason}",
                importer.display()
            ),
            ModuleError::Diagnostics { .. } => {
                formatter.write_str("module graph contains frontend diagnostics")
            }
        }
    }
}

impl Error for ModuleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ModuleError::Io { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Discover, parse and lower the complete static module graph rooted at `entry`.
///
/// `program` may already contain the Wren core/builtin metadata. All discovered
/// source modules are appended to the same Program so ImportModule can resolve
/// them by canonical module name after bytecode compilation.
pub fn compile_module_graph(
    program: &mut Program,
    entry: &Path,
) -> Result<CompiledModuleGraph, ModuleError> {
    let resolver = ModuleResolver::for_entry(entry)?;
    compile_module_graph_with_resolver(program, entry, &resolver)
}

pub fn compile_module_graph_with_resolver(
    program: &mut Program,
    entry: &Path,
    resolver: &ModuleResolver,
) -> Result<CompiledModuleGraph, ModuleError> {
    let entry = resolver.resolve_entry(entry)?;
    let mut modules = Vec::<ParsedModule>::new();
    let mut module_by_path = HashMap::<PathBuf, usize>::new();
    let mut pending = Vec::<usize>::new();
    let mut diagnostics = Vec::<ModuleDiagnosticSource>::new();

    let entry_index = load_parsed_module(
        &entry,
        &mut modules,
        &mut module_by_path,
        &mut pending,
        &mut diagnostics,
    )?;

    let mut cursor = 0usize;
    while cursor < pending.len() {
        let index = pending[cursor];
        cursor += 1;

        let importer_path = modules[index].path.clone();
        let imports = wren_front::module_imports(&modules[index].ast)
            .into_iter()
            .map(|import| {
                std::str::from_utf8(&import.module)
                    .map(str::to_owned)
                    .map_err(|_| ModuleError::InvalidSpecifier {
                        importer: importer_path.clone(),
                        specifier: String::from_utf8_lossy(&import.module).into_owned(),
                        reason: "module specifiers must be valid UTF-8".to_owned(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        for specifier in imports {
            if modules[index].import_map.contains_key(&specifier) {
                continue;
            }
            let resolved = resolver.resolve_import(&importer_path, &specifier)?;
            let canonical_name = resolved.canonical_name.clone();
            load_parsed_module(
                &resolved,
                &mut modules,
                &mut module_by_path,
                &mut pending,
                &mut diagnostics,
            )?;
            modules[index]
                .import_map
                .insert(specifier, canonical_name);
        }
    }

    // All modules are known before lowering. This is important for cycles: an
    // ImportModule operand can always name a module present in final metadata,
    // independent of lowering order.
    let mut lowered_entry = None;
    for (index, module) in modules.iter().enumerate() {
        let output = wren_lowerer::lower_module_with_options(
            program,
            &module.path.to_string_lossy(),
            &module.source,
            &module.canonical_name,
            &module.ast,
            LowerOptions {
                import_map: module.import_map.clone(),
                ..LowerOptions::default()
            },
        );

        let has_errors = output.has_errors();
        if !output.diagnostics.is_empty() {
            diagnostics.push(ModuleDiagnosticSource {
                source_name: module.path.to_string_lossy().into_owned(),
                source: module.source.clone(),
                diagnostics: output.diagnostics,
            });
        }
        if has_errors {
            return Err(ModuleError::Diagnostics { sources: diagnostics });
        }

        let Some(lowered) = output.lowered else {
            return Err(ModuleError::Diagnostics { sources: diagnostics });
        };
        if index == entry_index {
            lowered_entry = Some(lowered);
        }
    }

    let entry_lowered = lowered_entry.expect("entry module disappeared from discovered graph");
    Ok(CompiledModuleGraph {
        entry: entry_lowered,
        entry_name: modules[entry_index].canonical_name.clone(),
        module_count: modules.len(),
        diagnostics,
    })
}

fn load_parsed_module(
    resolved: &ResolvedModule,
    modules: &mut Vec<ParsedModule>,
    module_by_path: &mut HashMap<PathBuf, usize>,
    pending: &mut Vec<usize>,
    diagnostics: &mut Vec<ModuleDiagnosticSource>,
) -> Result<usize, ModuleError> {
    if let Some(index) = module_by_path.get(&resolved.path).copied() {
        return Ok(index);
    }

    let source = fs::read_to_string(&resolved.path).map_err(|error| ModuleError::Io {
        path: resolved.path.clone(),
        error,
    })?;
    let source_name = resolved.path.to_string_lossy().into_owned();
    let parsed = wren_front::parse(&source_name, &source);
    let has_errors = parsed.has_errors();
    if !parsed.diagnostics.is_empty() {
        diagnostics.push(ModuleDiagnosticSource {
            source_name: source_name.clone(),
            source: source.clone(),
            diagnostics: parsed.diagnostics,
        });
    }
    if has_errors {
        return Err(ModuleError::Diagnostics {
            sources: std::mem::take(diagnostics),
        });
    }

    // Insert before discovering its dependencies. A -> B -> A therefore finds
    // the existing path entry and terminates naturally.
    let index = modules.len();
    modules.push(ParsedModule {
        path: resolved.path.clone(),
        canonical_name: resolved.canonical_name.clone(),
        source,
        ast: parsed.module,
        import_map: HashMap::new(),
    });
    module_by_path.insert(resolved.path.clone(), index);
    pending.push(index);
    Ok(index)
}

fn emit_diagnostic_sources(sources: &[ModuleDiagnosticSource]) -> io::Result<()> {
    for source in sources {
        wren_front::emit_diagnostics(&source.source_name, &source.source, &source.diagnostics)?;
    }
    Ok(())
}

fn canonical_file(path: &Path) -> Result<PathBuf, ModuleError> {
    fs::canonicalize(path).map_err(|error| ModuleError::Io {
        path: path.to_owned(),
        error,
    })
}

fn canonical_directory(path: &Path) -> Option<PathBuf> {
    let canonical = fs::canonicalize(path).ok()?;
    canonical.is_dir().then_some(canonical)
}

fn discover_project_root(start: &Path) -> PathBuf {
    let canonical = fs::canonicalize(start).unwrap_or_else(|_| start.to_owned());
    for directory in canonical.ancestors() {
        if PROJECT_MARKERS.iter().any(|marker| directory.join(marker).exists()) {
            return directory.to_owned();
        }
    }
    canonical
}

fn extend_env_paths(out: &mut Vec<PathBuf>, variable: &str) {
    let Some(value) = env::var_os(variable) else {
        return;
    };
    out.extend(env::split_paths(&value));
}

fn dedupe_roots(roots: &mut Vec<PathBuf>) {
    let mut seen = HashSet::<OsString>::new();
    roots.retain(|root| {
        let normalized = fs::canonicalize(root).unwrap_or_else(|_| root.clone());
        seen.insert(normalized.into_os_string())
    });
}

fn validate_namespaced_tail(
    importer: &Path,
    specifier: &str,
    tail: &str,
) -> Result<(), ModuleError> {
    if tail.is_empty() {
        return Err(ModuleError::InvalidSpecifier {
            importer: importer.to_owned(),
            specifier: specifier.to_owned(),
            reason: "namespace module name cannot be empty".to_owned(),
        });
    }
    for component in Path::new(tail).components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => {
                return Err(ModuleError::InvalidSpecifier {
                    importer: importer.to_owned(),
                    specifier: specifier.to_owned(),
                    reason: "`..` is not allowed inside @lib/ or @std/ specifiers".to_owned(),
                });
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(ModuleError::InvalidSpecifier {
                    importer: importer.to_owned(),
                    specifier: specifier.to_owned(),
                    reason: "namespace module names must be relative".to_owned(),
                });
            }
        }
    }
    Ok(())
}

fn resolve_candidate_set(base: &Path, searched: &mut Vec<PathBuf>) -> Option<PathBuf> {
    for candidate in candidate_paths(base) {
        searched.push(candidate.clone());
        if !candidate.is_file() {
            continue;
        }
        if let Ok(canonical) = fs::canonicalize(&candidate) {
            return Some(canonical);
        }
    }
    None
}

fn candidate_paths(base: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::with_capacity(3);
    candidates.push(base.to_owned());

    if base.extension().is_none() {
        candidates.push(base.with_extension(SOURCE_EXTENSION));
    }

    candidates.push(base.join(format!("index.{SOURCE_EXTENSION}")));
    candidates
}

fn slash_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempTree {
        root: PathBuf,
    }

    impl TempTree {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = env::temp_dir().join(format!("wrenjit-modules-{}-{nonce}", std::process::id()));
            fs::create_dir_all(&root).unwrap();
            Self { root }
        }

        fn write(&self, relative: &str, source: &str) -> PathBuf {
            let path = self.root.join(relative);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&path, source).unwrap();
            path
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn resolves_relative_extension_and_directory_index() {
        let tree = TempTree::new();
        tree.write("wrenjit.toml", "");
        let entry = tree.write("src/main.wren", "");
        let direct = tree.write("src/util.wren", "");
        let index = tree.write("src/widgets/index.wren", "");
        let resolver = ModuleResolver::for_entry(&entry).unwrap();

        assert_eq!(resolver.resolve_import(&entry, "./util").unwrap().path, fs::canonicalize(direct).unwrap());
        assert_eq!(resolver.resolve_import(&entry, "./widgets").unwrap().path, fs::canonicalize(index).unwrap());
    }

    #[test]
    fn relative_and_namespaced_package_import_share_identity() {
        let tree = TempTree::new();
        tree.write("wrenjit.toml", "");
        let entry = tree.write("main.wren", "");
        let package = tree.write("packages/foo/index.wren", "");
        let child = tree.write("packages/foo/child.wren", "");

        let mut config = ModuleResolverConfig::for_entry(&entry).unwrap();
        config.package_roots.insert(0, tree.root.join("packages"));
        let resolver = ModuleResolver::new(config);

        let by_namespace = resolver.resolve_import(&entry, "@lib/foo/child").unwrap();
        let from_package = resolver.resolve_import(&package, "./child").unwrap();
        assert_eq!(by_namespace.path, fs::canonicalize(child).unwrap());
        assert_eq!(by_namespace.canonical_name, from_package.canonical_name);
        assert_eq!(by_namespace.canonical_name, "@lib/foo/child.wren");
    }

    #[test]
    fn discovers_a_complete_graph_and_records_exports() {
        let tree = TempTree::new();
        tree.write("wrenjit.toml", "");
        let entry = tree.write(
            "src/main.wren",
            r#"import "./dep" for Answer
var Seen = Answer
return Seen
"#,
        );
        tree.write("src/dep.wren", "var Answer = 42\n");

        let mut program = Program::new();
        let graph = compile_module_graph(&mut program, &entry).unwrap();
        assert_eq!(graph.module_count, 2);

        let dep = program
            .modules
            .iter()
            .find(|module| program.string(module.name) == Some("file:src/dep.wren"))
            .expect("dependency module metadata missing");
        assert!(dep
            .exports
            .iter()
            .any(|export| program.string(export.name) == Some("Answer")));

        // Exercise the whole path through bytecode and ImportVariable. This
        // catches mismatches between canonical import StringIds and runtime
        // module/export metadata that resolver-only tests cannot see.
        let bytecode = crate::wren_pipeline::compile_program(&mut program).unwrap();
        let mut vm = crate::wren_interpreter::Interpreter::new(&bytecode).unwrap();
        vm.set_jit_options(None);
        let crate::wren_interpreter::ExecutionResult::Returned(result) =
            vm.run_module(graph.entry.entry).unwrap()
        else {
            panic!("expected module to return");
        };
        assert_eq!(result.as_num(), Some(42.0));
    }

    #[test]
    fn imports_and_reexports_values_and_classes_with_aliases() {
        use crate::wren_builtin::{self, BuiltinHost};
        use crate::wren_interpreter::{ExecutionResult, Interpreter};

        let tree = TempTree::new();
        tree.write("wrenjit.toml", "");
        let entry = tree.write(
            "main.wren",
            r#"import "./bridge" for Answer as Result, Greeting, Calculator, Empty
import "./other" for Answer as Other
return "%(Greeting)|%(Result)|%(Calculator.add(20, 22))|%(Empty)|%(Other)"
"#,
        );
        tree.write(
            "dep.wren",
            r#"var Answer = 42
var Greeting = "hello"
var Empty = null
class Calculator {
  static add(a, b) {
    return a + b
  }
}
"#,
        );
        tree.write(
            "bridge.wren",
            "import \"./dep\" for Answer, Greeting, Calculator, Empty\n",
        );
        tree.write("other.wren", "var Answer = 7\n");

        let mut program = Program::new();
        let manifest = wren_builtin::install_hir_builtins(&mut program);
        let graph = compile_module_graph(&mut program, &entry).unwrap();
        assert_eq!(graph.module_count, 4);
        let bytecode = crate::wren_pipeline::compile_program(&mut program).unwrap();
        let host = Box::new(BuiltinHost::new(manifest.clone()));
        let mut vm = Interpreter::with_host(&bytecode, host).unwrap();
        wren_builtin::install_vm_builtins(&mut vm, &manifest);
        vm.set_jit_options(None);
        let ExecutionResult::Returned(result) = vm.run_module(graph.entry.entry).unwrap() else {
            panic!("expected module to return");
        };
        assert_eq!(vm.value_to_string(result), "hello|42|42|null|7");
    }

    #[test]
    fn importing_a_missing_export_still_fails() {
        let tree = TempTree::new();
        tree.write("wrenjit.toml", "");
        let entry = tree.write("main.wren", "import \"./dep\" for Missing\n");
        tree.write("dep.wren", "var Answer = 42\n");

        let mut program = Program::new();
        let graph = compile_module_graph(&mut program, &entry).unwrap();
        let bytecode = crate::wren_pipeline::compile_program(&mut program).unwrap();
        let mut vm = crate::wren_interpreter::Interpreter::new(&bytecode).unwrap();
        vm.set_jit_options(None);
        let error = vm.run_module(graph.entry.entry).unwrap_err();
        assert_eq!(error.message, "imported variable name is not bound");
    }

    #[test]
    fn cyclic_graph_is_discovered_once() {
        let tree = TempTree::new();
        tree.write("wrenjit.toml", "");
        let entry = tree.write("a.wren", r#"import "./b"
var A = 1
"#);
        tree.write("b.wren", r#"import "./a"
var B = 2
"#);

        let mut program = Program::new();
        let graph = compile_module_graph(&mut program, &entry).unwrap();
        assert_eq!(graph.module_count, 2);
        assert_eq!(program.modules.len(), 2);

        let bytecode = crate::wren_pipeline::compile_program(&mut program).unwrap();
        let mut vm = crate::wren_interpreter::Interpreter::new(&bytecode).unwrap();
        vm.set_jit_options(None);
        assert!(matches!(
            vm.run_module(graph.entry.entry).unwrap(),
            crate::wren_interpreter::ExecutionResult::Returned(_)
        ));
    }

    #[test]
    fn rejects_namespace_escape() {
        let tree = TempTree::new();
        tree.write("wrenjit.toml", "");
        let entry = tree.write("main.wren", "");
        let mut config = ModuleResolverConfig::for_entry(&entry).unwrap();
        config.package_roots.insert(0, tree.root.join("packages"));
        let resolver = ModuleResolver::new(config);
        assert!(matches!(
            resolver.resolve_import(&entry, "@lib/foo/../../secret"),
            Err(ModuleError::InvalidSpecifier { .. })
        ));
    }
}
