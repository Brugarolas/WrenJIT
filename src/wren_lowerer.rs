//! AST -> Wren IR lowering.
//!
//! This module consumes `wren_front`'s syntax tree and emits `wren_ir` SSA.
//! It deliberately keeps the source language dynamic: ordinary operators and
//! method calls become `Invoke`, while syntax-owned operations such as literal
//! list/map construction use dedicated IR nodes.
//!
//! Design goals:
//! - lexical name resolution before code generation;
//! - SSA locals by default;
//! - only captured *mutable* locals are heap cells;
//! - immutable captures are passed by value;
//! - block parameters, never Phi instructions, merge SSA state;
//! - loop-carried locals are explicit block parameters;
//! - method/closure capture chains are resolved before emission;
//! - Wren dispatch remains generic until later optimization/JIT passes;
//! - all lowering diagnostics are rendered with Ariadne.

#![allow(clippy::large_enum_variant)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::wren_front::{
    self as front, BinaryOp, ClassDecl, Diagnostic, DiagnosticLabel, DiagnosticSeverity, Expr,
    ExprKind, FunctionBody, FunctionBodyKind, Ident, ImportDecl, InvokeSyntax, LogicalOp,
    MethodDecl, MethodSignatureAst, ModuleAst, OperatorName, Span, Stmt, StmtKind, StringPart,
    UnaryOp,
};
use crate::wren_ir::{
    self as ir, Builder, BuiltinBinding, BuiltinId, CaptureMode, ClassId, ClassSet, FieldSlot,
    FuncId, FunctionKind, Inst, MethodKind, MethodSymbol, ModuleId, ModuleSlot, SourceSpan,
    StaticFieldSlot, Type, ValueId,
};

use ariadne::{Color, Label, Report, ReportKind, Source};
use hashbrown::{HashMap, HashSet};
use smallvec::SmallVec;
use std::io;

const MAX_WREN_ARITY: usize = 16;
const MAX_WREN_FIELDS: usize = u16::MAX as usize;

// =============================================================================
// Public API
// =============================================================================

#[derive(Clone, Debug)]
pub struct LowerOptions {
    /// Retained for source/API compatibility.
    ///
    /// Field slots are now class-local in WIR and relocated once at runtime
    /// class-link time, so dynamic/imported superclasses no longer make field
    /// layout ambiguous and this option no longer rejects them.
    pub require_static_super_layout_for_fields: bool,
    /// Warn for declarations that shadow another local in an outer lexical scope.
    pub warn_shadowing: bool,
}

impl Default for LowerOptions {
    fn default() -> Self {
        Self {
            require_static_super_layout_for_fields: true,
            warn_shadowing: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LoweredModule {
    pub module: ModuleId,
    pub entry: FuncId,
}

#[derive(Debug)]
pub struct LowerOutput {
    pub lowered: Option<LoweredModule>,
    pub diagnostics: Vec<Diagnostic>,
}

impl LowerOutput {
    #[inline]
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
    }

    /// Print lowering diagnostics through Ariadne.
    pub fn emit(&self, source_name: &str, source: &str) -> io::Result<()> {
        emit_lower_diagnostics(source_name, source, &self.diagnostics)
    }
}

pub struct OwnedLowerOutput {
    pub program: ir::Program,
    pub lowered: Option<LoweredModule>,
    pub diagnostics: Vec<Diagnostic>,
}

impl OwnedLowerOutput {
    #[inline]
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
    }

    pub fn emit(&self, source_name: &str, source: &str) -> io::Result<()> {
        emit_lower_diagnostics(source_name, source, &self.diagnostics)
    }
}

pub fn lower_module(
    program: &mut ir::Program,
    source_name: &str,
    source: &str,
    module_name: &str,
    module: &ModuleAst,
) -> LowerOutput {
    lower_module_with_options(
        program,
        source_name,
        source,
        module_name,
        module,
        LowerOptions::default(),
    )
}

pub fn lower_module_with_options(
    program: &mut ir::Program,
    source_name: &str,
    source: &str,
    module_name: &str,
    module: &ModuleAst,
    options: LowerOptions,
) -> LowerOutput {
    let mut lowerer = ModuleLowerer::new(program, source_name, source, module_name, module, options);
    let lowered = lowerer.lower();
    LowerOutput {
        lowered,
        diagnostics: lowerer.diagnostics,
    }
}

pub fn lower_new_program(
    source_name: &str,
    source: &str,
    module_name: &str,
    module: &ModuleAst,
) -> OwnedLowerOutput {
    let mut program = ir::Program::new();
    let output = lower_module(&mut program, source_name, source, module_name, module);
    OwnedLowerOutput {
        program,
        lowered: output.lowered,
        diagnostics: output.diagnostics,
    }
}

pub fn lower_and_report(
    source_name: &str,
    source: &str,
    module_name: &str,
    module: &ModuleAst,
) -> Result<(ir::Program, LoweredModule), ()> {
    let output = lower_new_program(source_name, source, module_name, module);
    let _ = output.emit(source_name, source);
    match output.lowered {
        Some(lowered) if !output.has_errors() => Ok((output.program, lowered)),
        _ => Err(()),
    }
}

pub fn emit_lower_diagnostics(
    source_name: &str,
    source: &str,
    diagnostics: &[Diagnostic],
) -> io::Result<()> {
    for diagnostic in diagnostics {
        let anchor = diagnostic
            .labels
            .first()
            .map(|label| byte_span_to_char_range(source, label.span))
            .unwrap_or(0..0);

        let kind = match diagnostic.severity {
            DiagnosticSeverity::Error => ReportKind::Error,
            DiagnosticSeverity::Warning => ReportKind::Warning,
        };

        let mut report = Report::build(kind, (source_name, anchor))
            .with_code(diagnostic.code)
            .with_message(diagnostic.message.clone());

        for diagnostic_label in &diagnostic.labels {
            let range = byte_span_to_char_range(source, diagnostic_label.span);
            let color = if diagnostic_label.primary {
                Color::Red
            } else {
                Color::Yellow
            };
            let mut label = Label::new((source_name, range)).with_color(color);
            if !diagnostic_label.message.is_empty() {
                label = label.with_message(diagnostic_label.message.clone());
            }
            report = report.with_label(label);
        }

        for note in &diagnostic.notes {
            report = report.with_note(note.clone());
        }

        report.finish().eprint((source_name, Source::from(source)))?;
    }
    Ok(())
}

fn byte_span_to_char_range(source: &str, span: Span) -> std::ops::Range<usize> {
    fn byte_to_char(source: &str, byte: usize) -> usize {
        let clamped = byte.min(source.len());
        source[..clamped].chars().count()
    }
    byte_to_char(source, span.start)..byte_to_char(source, span.end)
}

// =============================================================================
// Internal identities and semantic analysis
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct BindingId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FunctionNodeId(usize);

#[derive(Clone, Debug)]
struct BindingInfo {
    name: String,
    declaration: Span,
    owner: FunctionNodeId,
    captured: bool,
    mutated: bool,
    synthetic_this: bool,
}

#[derive(Clone, Debug)]
enum FunctionAst {
    Statements(Vec<Stmt>),
    Body(FunctionBody),
}

#[derive(Clone, Debug)]
struct FunctionNode {
    parent: Option<FunctionNodeId>,
    ast: FunctionAst,
    formal_params: Vec<Ident>,
    param_bindings: Vec<BindingId>,
    captures: Vec<BindingId>,
    capture_set: HashSet<BindingId>,
    this_binding: Option<BindingId>,
    closure_span: Option<Span>,
    is_async: bool,
}

#[derive(Clone, Debug)]
struct RootAnalysis {
    nodes: Vec<FunctionNode>,
    bindings: Vec<BindingInfo>,
    declaration_map: HashMap<Span, BindingId>,
    use_map: HashMap<Span, BindingId>,
    closure_map: HashMap<Span, FunctionNodeId>,
}

impl RootAnalysis {
    fn binding(&self, id: BindingId) -> &BindingInfo {
        &self.bindings[id.0 as usize]
    }
}

struct Analyzer {
    analysis: RootAnalysis,
    scopes: Vec<HashMap<String, BindingId>>,
    current_function: FunctionNodeId,
    function_stack: Vec<FunctionNodeId>,
    module_root: bool,
    diagnostics: Vec<Diagnostic>,
    warn_shadowing: bool,
}

impl Analyzer {
    fn module(module: &ModuleAst, warn_shadowing: bool) -> (RootAnalysis, Vec<Diagnostic>) {
        let root = FunctionNode {
            parent: None,
            ast: FunctionAst::Statements(module.statements.clone()),
            formal_params: Vec::new(),
            param_bindings: Vec::new(),
            captures: Vec::new(),
            capture_set: HashSet::new(),
            this_binding: None,
            closure_span: None,
            is_async: false,
        };
        let mut analyzer = Self {
            analysis: RootAnalysis {
                nodes: vec![root],
                bindings: Vec::new(),
                declaration_map: HashMap::new(),
                use_map: HashMap::new(),
                closure_map: HashMap::new(),
            },
            scopes: vec![HashMap::new()],
            current_function: FunctionNodeId(0),
            function_stack: vec![FunctionNodeId(0)],
            module_root: true,
            diagnostics: Vec::new(),
            warn_shadowing,
        };
        for statement in &module.statements {
            analyzer.visit_stmt(statement, true);
        }
        (analyzer.analysis, analyzer.diagnostics)
    }

    fn method(
        method: &MethodDecl,
        formal_params: Vec<Ident>,
        warn_shadowing: bool,
    ) -> (RootAnalysis, Vec<Diagnostic>) {
        let body = method.body.clone().unwrap_or(FunctionBody {
            params: Vec::new(),
            kind: FunctionBodyKind::Statements(Vec::new()),
            span: method.span,
        });
        let root = FunctionNode {
            parent: None,
            ast: FunctionAst::Body(body.clone()),
            formal_params: formal_params.clone(),
            param_bindings: Vec::new(),
            captures: Vec::new(),
            capture_set: HashSet::new(),
            this_binding: None,
            closure_span: None,
            is_async: method.is_async,
        };
        let mut analyzer = Self {
            analysis: RootAnalysis {
                nodes: vec![root],
                bindings: Vec::new(),
                declaration_map: HashMap::new(),
                use_map: HashMap::new(),
                closure_map: HashMap::new(),
            },
            scopes: vec![HashMap::new()],
            current_function: FunctionNodeId(0),
            function_stack: vec![FunctionNodeId(0)],
            module_root: false,
            diagnostics: Vec::new(),
            warn_shadowing,
        };

        let this_binding = analyzer.declare_synthetic_this(method.span);
        analyzer.analysis.nodes[0].this_binding = Some(this_binding);
        let mut param_bindings = Vec::new();
        for param in &formal_params {
            param_bindings.push(analyzer.declare(param));
        }
        analyzer.analysis.nodes[0].param_bindings = param_bindings;
        analyzer.visit_function_body_contents(&body);
        (analyzer.analysis, analyzer.diagnostics)
    }

    fn declare_synthetic_this(&mut self, span: Span) -> BindingId {
        let id = BindingId(self.analysis.bindings.len() as u32);
        self.analysis.bindings.push(BindingInfo {
            name: "this".to_owned(),
            declaration: Span::empty(span.start),
            owner: self.current_function,
            captured: false,
            mutated: false,
            synthetic_this: true,
        });
        id
    }

    fn declare(&mut self, ident: &Ident) -> BindingId {
        if self.warn_shadowing
            && self
                .scopes
                .iter()
                .rev()
                .skip(1)
                .any(|scope| scope.contains_key(&ident.value))
        {
            self.diagnostics.push(
                Diagnostic::warning(
                    "WREN-L001",
                    format!("local '{}' shadows an outer local", ident.value),
                    ident.span,
                )
                .with_note("Wren permits shadowing; this warning is informational."),
            );
        }

        let id = BindingId(self.analysis.bindings.len() as u32);
        self.analysis.bindings.push(BindingInfo {
            name: ident.value.clone(),
            declaration: ident.span,
            owner: self.current_function,
            captured: false,
            mutated: false,
            synthetic_this: false,
        });
        self.analysis.declaration_map.insert(ident.span, id);
        self.scopes
            .last_mut()
            .expect("analysis scope stack cannot be empty")
            .insert(ident.value.clone(), id);
        id
    }

    fn resolve(&self, name: &str) -> Option<BindingId> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }

    fn note_binding_use(&mut self, ident: &Ident, write: bool) {
        let Some(binding) = self.resolve(&ident.value) else {
            if is_implicit_receiver_name(&ident.value) {
                self.note_this_use();
            }
            return;
        };
        self.analysis.use_map.insert(ident.span, binding);
        if write {
            self.analysis.bindings[binding.0 as usize].mutated = true;
        }
        self.capture_if_needed(binding);
    }

    fn note_this_use(&mut self) {
        let this = self
            .function_stack
            .iter()
            .rev()
            .find_map(|function| self.analysis.nodes[function.0].this_binding);
        if let Some(binding) = this {
            self.capture_if_needed(binding);
        }
    }

    fn capture_if_needed(&mut self, binding: BindingId) {
        let owner = self.analysis.bindings[binding.0 as usize].owner;
        if owner == self.current_function {
            return;
        }
        self.analysis.bindings[binding.0 as usize].captured = true;

        let owner_pos = self
            .function_stack
            .iter()
            .position(|function| *function == owner)
            .expect("binding owner must be an active lexical function");
        for function in self.function_stack.iter().skip(owner_pos + 1).copied() {
            let node = &mut self.analysis.nodes[function.0];
            if node.capture_set.insert(binding) {
                node.captures.push(binding);
            }
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn visit_stmt(&mut self, stmt: &Stmt, module_level: bool) {
        match &stmt.kind {
            StmtKind::Variable { name, initializer } => {
                if module_level && self.module_root {
                    if let Some(initializer) = initializer {
                        self.visit_expr(initializer);
                    }
                } else {
                    self.declare(name);
                    if let Some(initializer) = initializer {
                        self.visit_expr(initializer);
                    }
                }
            }
            StmtKind::Class(class) => {
                if !(module_level && self.module_root) {
                    self.declare(&class.name);
                }
                if let Some(superclass) = &class.superclass {
                    self.visit_expr(superclass);
                }
                // Methods are compilation roots, not lexical closures of the
                // surrounding statement scope.
            }
            StmtKind::Import(import) => {
                if !(module_level && self.module_root) {
                    for binding in &import.bindings {
                        self.declare(&binding.local);
                    }
                }
            }
            StmtKind::Block(statements) => {
                self.push_scope();
                for statement in statements {
                    self.visit_stmt(statement, false);
                }
                self.pop_scope();
            }
            StmtKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.visit_expr(condition);
                self.visit_stmt(then_branch, false);
                if let Some(else_branch) = else_branch {
                    self.visit_stmt(else_branch, false);
                }
            }
            StmtKind::While { condition, body } => {
                self.visit_expr(condition);
                self.visit_stmt(body, false);
            }
            StmtKind::For {
                name,
                iterable,
                body,
            } => {
                self.visit_expr(iterable);
                self.push_scope();
                self.declare(name);
                self.visit_stmt(body, false);
                self.pop_scope();
            }
            StmtKind::Break | StmtKind::Continue => {}
            StmtKind::Return(value) => {
                if let Some(value) = value {
                    self.visit_expr(value);
                }
            }
            StmtKind::Expression(expr) => self.visit_expr(expr),
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::Null | ExprKind::Bool(_) | ExprKind::Number(_) | ExprKind::String(_) => {}
            ExprKind::InterpolatedString(parts) => {
                for part in parts {
                    if let StringPart::Expression(expr) = part {
                        self.visit_expr(expr);
                    }
                }
            }
            ExprKind::Name(name) => self.note_binding_use(name, false),
            ExprKind::Field(_) | ExprKind::StaticField(_) | ExprKind::This => self.note_this_use(),
            ExprKind::Super(super_expr) => {
                self.note_this_use();
                for arg in &super_expr.args {
                    self.visit_expr(arg);
                }
            }
            ExprKind::List(items) => {
                for item in items {
                    self.visit_expr(item);
                }
            }
            ExprKind::Map(entries) => {
                for entry in entries {
                    self.visit_expr(&entry.key);
                    self.visit_expr(&entry.value);
                }
            }
            ExprKind::Grouped(inner) => self.visit_expr(inner),
            ExprKind::Unary { operand, .. } => self.visit_expr(operand),
            ExprKind::Binary { left, right, .. } | ExprKind::Logical { left, right, .. } => {
                self.visit_expr(left);
                self.visit_expr(right);
            }
            ExprKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.visit_expr(condition);
                self.visit_expr(then_expr);
                self.visit_expr(else_expr);
            }
            ExprKind::Assign { target, value } => {
                self.visit_assignment_target(target);
                self.visit_expr(value);
            }
            ExprKind::Invoke(invoke) => {
                if let Some(receiver) = &invoke.receiver {
                    self.visit_expr(receiver);
                } else {
                    self.note_this_use();
                }
                for arg in &invoke.args {
                    self.visit_expr(arg);
                }
            }
            ExprKind::Subscript { receiver, args } => {
                self.visit_expr(receiver);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            ExprKind::Await(value) => self.visit_expr(value),
            ExprKind::Closure(body) => self.visit_closure(body, false),
            ExprKind::AsyncClosure(body) => self.visit_closure(body, true),
        }
    }

    fn visit_assignment_target(&mut self, target: &Expr) {
        match &target.kind {
            ExprKind::Name(name) => self.note_binding_use(name, true),
            ExprKind::Field(_) | ExprKind::StaticField(_) => self.note_this_use(),
            ExprKind::Invoke(invoke) => {
                if let Some(receiver) = &invoke.receiver {
                    self.visit_expr(receiver);
                } else {
                    self.note_this_use();
                }
            }
            ExprKind::Subscript { receiver, args } => {
                self.visit_expr(receiver);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            _ => self.visit_expr(target),
        }
    }

    fn visit_closure(&mut self, body: &FunctionBody, is_async: bool) {
        let parent = self.current_function;
        let id = FunctionNodeId(self.analysis.nodes.len());
        self.analysis.nodes.push(FunctionNode {
            parent: Some(parent),
            ast: FunctionAst::Body(body.clone()),
            formal_params: body.params.clone(),
            param_bindings: Vec::new(),
            captures: Vec::new(),
            capture_set: HashSet::new(),
            this_binding: self.analysis.nodes[parent.0].this_binding,
            closure_span: Some(body.span),
            is_async,
        });
        self.analysis.closure_map.insert(body.span, id);

        self.current_function = id;
        self.function_stack.push(id);
        self.push_scope();

        let mut params = Vec::new();
        for param in &body.params {
            params.push(self.declare(param));
        }
        self.analysis.nodes[id.0].param_bindings = params;
        self.visit_function_body_contents(body);

        self.pop_scope();
        self.function_stack.pop();
        self.current_function = parent;
    }

    fn visit_function_body_contents(&mut self, body: &FunctionBody) {
        match &body.kind {
            FunctionBodyKind::Expression(expr) => self.visit_expr(expr),
            FunctionBodyKind::Statements(statements) => {
                for statement in statements {
                    self.visit_stmt(statement, false);
                }
            }
        }
    }
}

// =============================================================================
// Module/class planning
// =============================================================================

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct MethodKey {
    name: String,
    kind: MethodKind,
    arity: u8,
}

#[derive(Clone, Debug, Default)]
struct SymbolTable {
    methods: HashMap<MethodKey, MethodSymbol>,
}

impl SymbolTable {
    fn get(&self, name: &str, kind: MethodKind, arity: usize) -> Option<MethodSymbol> {
        let arity = u8::try_from(arity).ok()?;
        self.methods
            .get(&MethodKey {
                name: name.to_owned(),
                kind,
                arity,
            })
            .copied()
    }

    fn expect(&self, name: &str, kind: MethodKind, arity: usize) -> MethodSymbol {
        self.get(name, kind, arity)
            .unwrap_or_else(|| panic!("method symbol was not preinterned: {name}/{arity:?}"))
    }
}

#[derive(Clone, Debug)]
struct MethodPlan {
    symbol: MethodSymbol,
    function: Option<FuncId>,
    foreign: Option<ir::ForeignFnId>,
    bind_static: bool,
}

#[derive(Clone, Debug)]
struct ClassPlan {
    id: ClassId,
    /// Hidden module root for this lexical class, independent of user rebinding.
    class_slot: ModuleSlot,
    name: String,
    declaration_span: Span,
    superclass: Option<ClassId>,
    /// Instance fields declared directly by this class. Slots always start at
    /// zero; runtime class linking adds the inherited field base once.
    fields: HashMap<String, FieldSlot>,
    static_fields: HashMap<String, StaticFieldSlot>,
    methods: Vec<MethodPlan>,
}

#[derive(Clone, Debug)]
struct ModuleLayout {
    slots: HashMap<String, ModuleSlot>,
}

impl ModuleLayout {
    fn get(&self, name: &str) -> Option<ModuleSlot> {
        self.slots.get(name).copied()
    }
}

struct ModuleLowerer<'p, 's, 'a> {
    program: &'p mut ir::Program,
    source_name: &'s str,
    source: &'s str,
    module_name: &'s str,
    ast: &'a ModuleAst,
    options: LowerOptions,
    diagnostics: Vec<Diagnostic>,
    source_file: ir::StringId,
    module_id: Option<ModuleId>,
    module_layout: ModuleLayout,
    symbols: SymbolTable,
    class_plans: HashMap<Span, ClassPlan>,
    top_level_classes: HashMap<String, ClassId>,
    string_ids: HashMap<String, ir::StringId>,
    closure_functions: HashMap<(usize, usize), FuncId>,
    method_functions: HashMap<Span, FuncId>,
}

impl<'p, 's, 'a> ModuleLowerer<'p, 's, 'a> {
    fn new(
        program: &'p mut ir::Program,
        source_name: &'s str,
        source: &'s str,
        module_name: &'s str,
        ast: &'a ModuleAst,
        options: LowerOptions,
    ) -> Self {
        let source_file = program.intern_string(source_name);
        Self {
            program,
            source_name,
            source,
            module_name,
            ast,
            options,
            diagnostics: Vec::new(),
            source_file,
            module_id: None,
            module_layout: ModuleLayout {
                slots: HashMap::new(),
            },
            symbols: SymbolTable::default(),
            class_plans: HashMap::new(),
            top_level_classes: HashMap::new(),
            string_ids: HashMap::new(),
            closure_functions: HashMap::new(),
            method_functions: HashMap::new(),
        }
    }

    fn lower(&mut self) -> Option<LoweredModule> {
        self.plan_module_slots();
        let module_id = self
            .program
            .add_module(self.module_name, self.module_layout.slots.len() as u32);
        self.module_id = Some(module_id);
        let builtin_bindings = BuiltinId::ALL
            .into_iter()
            .filter_map(|builtin| {
                self.module_layout
                    .get(builtin.name())
                    .map(|slot| BuiltinBinding { slot, builtin })
            })
            .collect();
        self.program
            .set_module_builtin_bindings(module_id, builtin_bindings);

        self.plan_classes(module_id);
        self.collect_and_intern_symbols();
        self.preintern_strings();
        self.compile_methods(module_id);

        let (analysis, analyzer_diagnostics) = Analyzer::module(self.ast, self.options.warn_shadowing);
        self.diagnostics.extend(analyzer_diagnostics);
        self.compile_closure_descendants(module_id, &analysis, FunctionNodeId(0), None);

        let spec = FunctionSpec {
            name: format!("<module {}>", self.module_name),
            kind: FunctionKind::ModuleBody,
            class: None,
            method_symbol: None,
            constructor: false,
            static_method: false,
            is_async: false,
        };
        let entry = self.compile_function_node(module_id, &analysis, FunctionNodeId(0), spec)?;

        if self.has_errors() {
            None
        } else {
            Some(LoweredModule {
                module: module_id,
                entry,
            })
        }
    }

    fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
    }

    fn error(&mut self, code: &'static str, message: impl Into<String>, span: Span) {
        self.diagnostics.push(Diagnostic::error(code, message, span));
    }

    fn warning(&mut self, code: &'static str, message: impl Into<String>, span: Span) {
        self.diagnostics
            .push(Diagnostic::warning(code, message, span));
    }

    fn plan_module_slots(&mut self) {
        let mut declarations: Vec<(String, Span)> = Vec::new();
        for statement in &self.ast.statements {
            match &statement.kind {
                StmtKind::Variable { name, .. } => {
                    declarations.push((name.value.clone(), name.span));
                }
                StmtKind::Class(class) => {
                    declarations.push((class.name.value.clone(), class.name.span));
                }
                StmtKind::Import(import) => {
                    declarations.extend(
                        import
                            .bindings
                            .iter()
                            .map(|binding| (binding.local.value.clone(), binding.local.span)),
                    );
                }
                _ => {}
            }
        }

        let mut next = 0u32;

        // Core globals are ordinary module slots after lowering. This keeps the
        // runtime hot path identical to user globals: one indexed load, no name
        // lookup and no dedicated LoadBuiltin opcode.
        for builtin in BuiltinId::ALL {
            self.module_layout
                .slots
                .insert(builtin.name().to_owned(), ModuleSlot(next));
            next += 1;
        }

        // Keep lexical class objects accessible from inherited instance methods
        // and nested closures, even when the user-facing class name is rebound.
        let mut classes = Vec::new();
        collect_class_declarations(&self.ast.statements, &mut classes);
        for (class, _) in classes {
            self.module_layout.slots.insert(
                format!("<class:{}>", class.name.span.start), ModuleSlot(next),
            );
            next += 1;
        }

        for (name, span) in declarations {
            self.insert_module_slot(&name, span, &mut next);
        }
    }

    fn insert_module_slot(&mut self, name: &str, span: Span, next: &mut u32) {
        if self.module_layout.slots.contains_key(name) {
            self.error(
                "WREN-L002",
                format!("module variable '{name}' is already declared"),
                span,
            );
            return;
        }
        self.module_layout
            .slots
            .insert(name.to_owned(), ModuleSlot(*next));
        *next += 1;
    }

    fn plan_classes(&mut self, module: ModuleId) {
        let mut declaration_refs = Vec::new();
        collect_class_declarations(&self.ast.statements, &mut declaration_refs);
        // The AST is immutable for the entire Lowerer lifetime. Keep stable raw
        // pointers instead of deep-cloning ClassDecl (including methods/bodies)
        // merely to work around field-disjoint mutable borrows of `self` below.
        // No pointer escapes this planning phase.
        let declarations: SmallVec<[(*const ClassDecl, bool); 8]> = declaration_refs
            .into_iter()
            .map(|(class, top)| (class as *const ClassDecl, top))
            .collect();

        // First reserve every runtime class id so forward class references can be
        // resolved during the second pass.
        for &(class_ptr, top_level) in &declarations {
            let class = unsafe { &*class_ptr };
            let id = self.program.add_class(
                module,
                &class.name.value,
                None,
                0,
                0,
                class.is_foreign,
            );
            if top_level {
                self.top_level_classes
                    .insert(class.name.value.clone(), id);
            }
            self.class_plans.insert(
                class.name.span,
                ClassPlan {
                    id,
                    class_slot: self.module_layout.get(&format!("<class:{}>", class.name.span.start)).unwrap(),
                    name: class.name.value.clone(),
                    declaration_span: class.name.span,
                    superclass: None,
                    fields: HashMap::new(),
                    static_fields: HashMap::new(),
                    methods: Vec::new(),
                },
            );
        }

        // Field names are lexical to the class and can appear in nested closures
        // inside methods, so gather the complete set before assigning slots.
        let mut local_instance_fields: HashMap<Span, SmallVec<[&str; 8]>> = HashMap::new();
        let mut local_static_fields: HashMap<Span, SmallVec<[&str; 8]>> = HashMap::new();
        for &(class_ptr, _) in &declarations {
            let class = unsafe { &*class_ptr };
            let mut instance = SmallVec::<[&str; 8]>::new();
            let mut statics = SmallVec::<[&str; 8]>::new();
            for method in &class.methods {
                if let Some(body) = &method.body {
                    collect_fields_in_body(body, &mut instance, &mut statics);
                }
            }
            dedup_preserve_order(&mut instance);
            dedup_preserve_order(&mut statics);
            local_instance_fields.insert(class.name.span, instance);
            local_static_fields.insert(class.name.span, statics);
        }

        // Resolve statically-known superclass identities for metadata/cycle
        // diagnostics. Instance field slots remain local to the declaring class;
        // runtime class linking relocates them after the actual superclass is
        // known, so imported/dynamic superclasses are fully supported.
        let mut resolving = HashSet::new();
        let mut resolved = HashSet::new();
        for &(class_ptr, _) in &declarations {
            let class = unsafe { &*class_ptr };
            self.resolve_class_layout(
                class,
                &local_instance_fields,
                &local_static_fields,
                &mut resolving,
                &mut resolved,
            );
        }
    }

    fn resolve_class_layout(
        &mut self,
        class: &ClassDecl,
        local_instance_fields: &HashMap<Span, SmallVec<[&str; 8]>>,
        local_static_fields: &HashMap<Span, SmallVec<[&str; 8]>>,
        resolving: &mut HashSet<Span>,
        resolved: &mut HashSet<Span>,
    ) {
        let key = class.name.span;
        if resolved.contains(&key) {
            return;
        }
        if !resolving.insert(key) {
            self.error(
                "WREN-L003",
                "cyclic statically-known superclass relationship",
                class.name.span,
            );
            return;
        }

        let static_super = class.superclass.as_ref().and_then(|expr| match &expr.kind {
            ExprKind::Name(name) => self.top_level_classes.get(&name.value).copied(),
            _ => None,
        });

        // Recurse only to preserve the useful compile-time cycle diagnostic.
        // Slot numbering itself is independent of the superclass layout.
        if let Some(super_id) = static_super {
            if let Some((super_decl, _)) = find_class_by_id(self.program, super_id, self.ast) {
                // `self.ast` is immutable while lowering. Capture the stable node
                // address before mutably recursing through `self` to avoid cloning
                // the complete superclass AST subtree.
                let super_decl = super_decl as *const ClassDecl;
                self.resolve_class_layout(
                    unsafe { &*super_decl },
                    local_instance_fields,
                    local_static_fields,
                    resolving,
                    resolved,
                );
            }
        }

        let locals = local_instance_fields.get(&key).cloned().unwrap_or_default();
        if locals.len() > MAX_WREN_FIELDS {
            self.error(
                "WREN-L005",
                "class contains too many instance fields",
                class.name.span,
            );
        }

        let statics = local_static_fields.get(&key).cloned().unwrap_or_default();
        if statics.len() > MAX_WREN_FIELDS {
            self.error(
                "WREN-L006",
                "class contains too many static fields",
                class.name.span,
            );
        }

        // Fixed-layout slots are local to the declaring class here. The runtime
        // turns them into absolute slots:
        //
        //   absolute = superclass.total_instance_fields + local_slot
        //
        // once, when CreateClass executes.
        let mut fields = HashMap::new();
        for (index, name) in locals.into_iter().enumerate() {
            fields.insert(name.to_owned(), FieldSlot(index as u16));
        }

        let mut static_fields = HashMap::new();
        for (index, name) in statics.into_iter().enumerate() {
            static_fields.insert(name.to_owned(), StaticFieldSlot(index as u16));
        }

        let id = self.class_plans[&key].id;
        if let Some(info) = self.program.classes.iter_mut().find(|info| info.id == id) {
            info.superclass = static_super;
            info.own_instance_field_count = fields.len() as u16;
            info.static_field_count = static_fields.len() as u16;
        }

        let plan = self.class_plans.get_mut(&key).unwrap();
        plan.superclass = static_super;
        plan.fields = fields;
        plan.static_fields = static_fields;

        resolving.remove(&key);
        resolved.insert(key);
    }

    fn collect_and_intern_symbols(&mut self) {
        let mut keys = HashSet::new();
        collect_method_keys_in_statements(&self.ast.statements, &mut keys);

        // Compiler-generated protocol calls.
        keys.insert(MethodKey {
            name: "iterate".to_owned(),
            kind: MethodKind::Method,
            arity: 1,
        });
        keys.insert(MethodKey {
            name: "iteratorValue".to_owned(),
            kind: MethodKind::Method,
            arity: 1,
        });
        keys.insert(MethodKey {
            name: "toString".to_owned(),
            kind: MethodKind::Getter,
            arity: 0,
        });

        for key in keys {
            let symbol = self.program.intern_method(&key.name, key.kind, key.arity);
            self.symbols.methods.insert(key, symbol);
        }
    }

    fn preintern_strings(&mut self) {
        self.string_ids.insert(self.source_name.to_owned(), self.program.intern_string(self.source_name));
        self.string_ids.insert(self.module_name.to_owned(), self.program.intern_string(self.module_name));
        let mut strings = Vec::new();
        collect_strings_in_statements(&self.ast.statements, &mut strings);
        for bytes in strings {
            if let Ok(text) = std::str::from_utf8(&bytes) {
                let id = self.program.intern_string(text);
                self.string_ids.insert(text.to_owned(), id);
            }
        }
        let mut imports = Vec::new();
        collect_import_strings(&self.ast.statements, &mut imports);
        for string in imports {
            let id = self.program.intern_string(&string);
            self.string_ids.insert(string, id);
        }
    }

    fn compile_methods(&mut self, module: ModuleId) {
        let mut class_refs = Vec::new();
        collect_class_declarations(&self.ast.statements, &mut class_refs);
        let classes: Vec<ClassDecl> = class_refs
            .into_iter()
            .map(|(class, _)| class.clone())
            .collect();
        for class in classes {
            let class_plan = self.class_plans[&class.name.span].clone();
            let mut plans = Vec::new();
            for method in &class.methods {
                let Some((name, kind, arity)) = method_signature_key(&method.signature) else {
                    continue;
                };
                if arity > MAX_WREN_ARITY {
                    self.error(
                        "WREN-L007",
                        format!("Wren methods may have at most {MAX_WREN_ARITY} parameters"),
                        method.span,
                    );
                    continue;
                }
                let symbol = self.symbols.expect(&name, kind, arity);
                let bind_static = method.is_static || method.is_constructor;

                if method.is_foreign || method.body.is_none() {
                    let foreign = self.program.add_foreign_function(
                        self.module_name,
                        &class.name.value,
                        symbol,
                        bind_static,
                    );
                    plans.push(MethodPlan {
                        symbol,
                        function: None,
                        foreign: Some(foreign),
                        bind_static,
                    });
                    continue;
                }

                let params = method_formal_params(&method.signature);
                // Constructor bodies are private instance initializers. Their
                // public static factories allocate once before calling them.
                let body_symbol = if method.is_constructor {
                    let key = MethodKey {
                        name: format!("<init>{name}"), kind: MethodKind::Method, arity: arity as u8,
                    };
                    let symbol = self.program.intern_method(&key.name, key.kind, key.arity);
                    self.symbols.methods.insert(key, symbol);
                    symbol
                } else {
                    symbol
                };
                let (analysis, diagnostics) =
                    Analyzer::method(method, params.clone(), self.options.warn_shadowing);
                self.diagnostics.extend(diagnostics);

                self.compile_closure_descendants(
                    module,
                    &analysis,
                    FunctionNodeId(0),
                    Some(MethodContextSpec {
                        class: class_plan.id,
                        method_symbol: body_symbol,
                        constructor: method.is_constructor,
                        static_method: method.is_static,
                        fields: class_plan.fields.clone(),
                        static_fields: class_plan.static_fields.clone(),
                    }),
                );

                let function_kind = if method.is_constructor {
                    FunctionKind::Constructor
                } else if method.is_static {
                    FunctionKind::StaticMethod
                } else {
                    FunctionKind::Method
                };
                let spec = FunctionSpec {
                    name: format!("{}.{}", class.name.value, display_method_signature(&method.signature)),
                    kind: function_kind,
                    class: Some(class_plan.id),
                    method_symbol: Some(body_symbol),
                    constructor: method.is_constructor,
                    static_method: method.is_static,
                    is_async: method.is_async,
                };
                if let Some(function) = self.compile_function_node(module, &analysis, FunctionNodeId(0), spec) {
                    self.method_functions.insert(method.span, function);
                    if method.is_constructor {
                        plans.push(MethodPlan {
                            symbol: body_symbol,
                            function: Some(function),
                            foreign: None,
                            bind_static: false,
                        });
                        let factory = self.compile_constructor_factory(module, &class_plan, &name, arity, Some(function));
                        plans.push(MethodPlan {
                            symbol, function: Some(factory), foreign: None, bind_static: true,
                        });
                        continue;
                    }
                    plans.push(MethodPlan {
                        symbol,
                        function: Some(function),
                        foreign: None,
                        bind_static,
                    });
                }
            }
            // WrenJIT convenience: a class with no declared constructors gets
            // an empty new(). Do not replace an explicitly defined static new().
            if !class.is_foreign && !class.methods.iter().any(|method| method.is_constructor)
                && !class.methods.iter().any(|method| method.is_static
                    && method_signature_key(&method.signature) == Some(("new".to_owned(), MethodKind::Method, 0)))
            {
                let symbol = self.program.intern_method("new", MethodKind::Method, 0);
                let factory = self.compile_constructor_factory(module, &class_plan, "new", 0, None);
                plans.push(MethodPlan {
                    symbol, function: Some(factory), foreign: None, bind_static: true,
                });
            }
            self.class_plans.get_mut(&class.name.span).unwrap().methods = plans;
        }
    }

    fn compile_constructor_factory(
        &mut self,
        module: ModuleId,
        class: &ClassPlan,
        name: &str,
        arity: usize,
        initializer: Option<FuncId>,
    ) -> FuncId {
        let mut params = vec![Type::Class(ClassSet::singleton(class.id))];
        params.extend((0..arity).map(|_| Type::Top));
        let mut builder = Builder::new(
            self.program, module, format!("{}.{name}<factory>", class.name),
            FunctionKind::StaticMethod, params, Type::Instance(ClassSet::singleton(class.id)),
        );
        let instance = builder.alloc_instance(class.id);
        if let Some(callee) = initializer {
            let mut args = vec![instance];
            args.extend((1..=arity).map(|index| builder.param(index)));
            builder.emit1(Inst::CallDirect { conv: ir::CallConv::Wren, callee, args }, Type::Top);
        }
        builder.ret(instance);
        builder.finish()
    }

    fn compile_closure_descendants(
        &mut self,
        module: ModuleId,
        analysis: &RootAnalysis,
        parent: FunctionNodeId,
        method_context: Option<MethodContextSpec>,
    ) {
        let children: Vec<_> = analysis
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                (node.parent == Some(parent)).then_some(FunctionNodeId(index))
            })
            .collect();

        for child in children {
            self.compile_closure_descendants(module, analysis, child, method_context.clone());
            let span = analysis.nodes[child.0]
                .closure_span
                .expect("non-root function nodes are closures");
            let spec = FunctionSpec {
                name: format!("<closure@{}>", span.start),
                kind: FunctionKind::Closure,
                class: method_context.as_ref().map(|context| context.class),
                method_symbol: method_context.as_ref().map(|context| context.method_symbol),
                constructor: false,
                static_method: method_context
                    .as_ref()
                    .map(|context| context.static_method)
                    .unwrap_or(false),
                is_async: analysis.nodes[child.0].is_async,
            };
            if let Some(function) = self.compile_function_node(module, analysis, child, spec) {
                self.closure_functions
                    .insert((analysis as *const _ as usize, child.0), function);
            }
        }
    }

    fn compile_function_node(
        &mut self,
        module: ModuleId,
        analysis: &RootAnalysis,
        node_id: FunctionNodeId,
        spec: FunctionSpec,
    ) -> Option<FuncId> {
        let node = &analysis.nodes[node_id.0];
        let has_receiver = spec.class.is_some() && spec.kind != FunctionKind::Closure;
        let receiver_type = if has_receiver {
            spec.class.map(|class| {
                if spec.static_method {
                    Type::Class(ClassSet::singleton(class))
                } else {
                    Type::Instance(ClassSet::singleton(class))
                }
            })
        } else {
            None
        };
        let mut param_types = Vec::new();
        if let Some(receiver_type) = receiver_type {
            param_types.push(receiver_type);
        }
        param_types.extend((0..node.formal_params.len()).map(|_| Type::Top));

        let mut builder = Builder::new(
            self.program,
            module,
            &spec.name,
            spec.kind,
            param_types,
            Type::Top,
        );
        builder.set_async(spec.is_async);
        if let Some(class) = spec.class {
            // Methods and closures nested inside them use field slots local to
            // this declaring class. The runtime relocates them exactly once when
            // CreateClass knows the actual superclass layout.
            builder.set_field_owner(class);
        }

        // Captures must be declared before *any* emitted instruction.
        let mut capture_values = HashMap::new();
        for binding in &node.captures {
            let info = analysis.binding(*binding);
            let mode = if info.mutated {
                CaptureMode::ByCell
            } else {
                CaptureMode::ByValue
            };
            let ty = if mode == CaptureMode::ByCell {
                Type::Cell(Box::new(Type::Top))
            } else {
                Type::Top
            };
            let value = builder.add_capture(ty, mode);
            capture_values.insert(*binding, Storage::from_capture(value, mode));
        }

        let method_context = spec.class.map(|class| {
            let plan = self
                .class_plans
                .values()
                .find(|plan| plan.id == class)
                .cloned()
                .expect("class plan must exist for method function");
            MethodContext {
                class,
                method_symbol: spec.method_symbol,
                constructor: spec.constructor,
                static_method: spec.static_method,
                fields: plan.fields,
                static_fields: plan.static_fields,
            }
        });

        let closure_ids: HashMap<FunctionNodeId, FuncId> = analysis
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, _)| {
                let id = FunctionNodeId(index);
                self.closure_functions
                    .get(&(analysis as *const _ as usize, id.0))
                    .copied()
                    .map(|function| (id, function))
            })
            .collect();

        let context = FunctionCompilerContext {
            source_name: self.source_name,
            source: self.source,
            source_file: self.source_file,
            module,
            module_layout: &self.module_layout,
            symbols: &self.symbols,
            class_plans: &self.class_plans,
            string_ids: &self.string_ids,
            analysis,
            node_id,
            closure_ids,
            method_context,
            has_receiver,
            allows_await: node.is_async || (node_id.0 == 0 && spec.kind == FunctionKind::ModuleBody),
        };
        let diagnostics = &mut self.diagnostics;
        let mut compiler = FunctionCompiler::new(builder, context, diagnostics, capture_values);
        compiler.initialize_parameters();
        compiler.lower_body();
        let function_id = compiler.finish();

        if let Some(function) = self.program.function(function_id) {
            if let Err(errors) = ir::verify_function(function) {
                let span = function_node_span(&analysis.nodes[node_id.0]);
                for error in errors {
                    self.diagnostics.push(
                        Diagnostic::error(
                            "WREN-L900",
                            format!("generated invalid WIR: {error}"),
                            span,
                        )
                        .with_note("This indicates a lowering bug, not a Wren source error."),
                    );
                }
            }
        }
        Some(function_id)
    }
}

// =============================================================================
// Function lowering
// =============================================================================

#[derive(Clone, Debug)]
struct FunctionSpec {
    name: String,
    kind: FunctionKind,
    class: Option<ClassId>,
    method_symbol: Option<MethodSymbol>,
    constructor: bool,
    static_method: bool,
    is_async: bool,
}

#[derive(Clone, Debug)]
struct MethodContextSpec {
    class: ClassId,
    method_symbol: MethodSymbol,
    constructor: bool,
    static_method: bool,
    fields: HashMap<String, FieldSlot>,
    static_fields: HashMap<String, StaticFieldSlot>,
}

#[derive(Clone, Debug)]
struct MethodContext {
    class: ClassId,
    method_symbol: Option<MethodSymbol>,
    constructor: bool,
    static_method: bool,
    fields: HashMap<String, FieldSlot>,
    static_fields: HashMap<String, StaticFieldSlot>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Storage {
    Value(ValueId),
    Cell(ValueId),
}

impl Storage {
    fn from_capture(value: ValueId, mode: CaptureMode) -> Self {
        match mode {
            CaptureMode::ByValue => Storage::Value(value),
            CaptureMode::ByCell => Storage::Cell(value),
        }
    }
}

#[derive(Clone, Debug)]
struct LoopContext {
    continue_target: ir::BlockId,
    break_target: ir::BlockId,
    carried: Vec<BindingId>,
    continue_prefix: Vec<ValueId>,
}

struct FunctionCompilerContext<'a> {
    source_name: &'a str,
    source: &'a str,
    source_file: ir::StringId,
    module: ModuleId,
    module_layout: &'a ModuleLayout,
    symbols: &'a SymbolTable,
    class_plans: &'a HashMap<Span, ClassPlan>,
    string_ids: &'a HashMap<String, ir::StringId>,
    analysis: &'a RootAnalysis,
    node_id: FunctionNodeId,
    closure_ids: HashMap<FunctionNodeId, FuncId>,
    method_context: Option<MethodContext>,
    has_receiver: bool,
    allows_await: bool,
}

struct FunctionCompiler<'p, 'a, 'd> {
    builder: Builder<'p>,
    context: FunctionCompilerContext<'a>,
    diagnostics: &'d mut Vec<Diagnostic>,
    env: HashMap<BindingId, Storage>,
    loops: Vec<LoopContext>,
    constructor_this: Option<ValueId>,
}

impl<'p, 'a, 'd> FunctionCompiler<'p, 'a, 'd> {
    fn new(
        builder: Builder<'p>,
        context: FunctionCompilerContext<'a>,
        diagnostics: &'d mut Vec<Diagnostic>,
        env: HashMap<BindingId, Storage>,
    ) -> Self {
        Self {
            builder,
            context,
            diagnostics,
            env,
            loops: Vec::new(),
            constructor_this: None,
        }
    }

    fn initialize_parameters(&mut self) {
        let node = &self.context.analysis.nodes[self.context.node_id.0];
        let mut param_offset = 0usize;

        if self.context.has_receiver {
            if let Some(method) = &self.context.method_context {
                let receiver = self.builder.param(0);
                param_offset = 1;
                if let Some(this_binding) = node.this_binding {
                    let this_value = if method.constructor {
                        self.constructor_this = Some(receiver);
                        receiver
                    } else {
                        receiver
                    };
                    self.env.insert(this_binding, Storage::Value(this_value));
                }
            }
        }

        for (index, binding) in node.param_bindings.iter().copied().enumerate() {
            let value = self.builder.param(param_offset + index);
            self.bind_initial(binding, value);
        }
    }

    fn bind_initial(&mut self, binding: BindingId, value: ValueId) {
        let info = self.context.analysis.binding(binding);
        if info.captured && info.mutated {
            let cell = self.builder.alloc_cell(value, Type::Top);
            self.env.insert(binding, Storage::Cell(cell));
        } else {
            self.env.insert(binding, Storage::Value(value));
        }
    }

    fn lower_body(&mut self) {
        let node = self.context.analysis.nodes[self.context.node_id.0].clone();
        match node.ast {
            FunctionAst::Statements(statements) => {
                self.lower_statements(&statements, self.context.node_id.0 == 0 && self.context.method_context.is_none());
                self.emit_default_return();
            }
            FunctionAst::Body(body) => match body.kind {
                FunctionBodyKind::Expression(expr) => {
                    let value = self.lower_expr(&expr);
                    if !self.current_terminated() {
                        if let Some(instance) = self.constructor_this {
                            self.builder.ret(instance);
                        } else {
                            self.builder.ret(value);
                        }
                    }
                }
                FunctionBodyKind::Statements(statements) => {
                    self.lower_statements(&statements, false);
                    self.emit_default_return();
                }
            },
        }
    }

    fn emit_default_return(&mut self) {
        if self.current_terminated() {
            return;
        }
        if let Some(instance) = self.constructor_this {
            self.builder.ret(instance);
        } else {
            let null = self.builder.const_null();
            self.builder.ret(null);
        }
    }

    fn finish(self) -> FuncId {
        self.builder.finish()
    }

    fn current_terminated(&self) -> bool {
        self.builder.block_terminated(self.builder.current_block())
    }

    fn set_source(&mut self, span: Span) {
        self.builder
            .set_source(Some(source_span(self.context.source, self.context.source_file, span)));
    }

    fn error(&mut self, code: &'static str, message: impl Into<String>, span: Span) {
        self.diagnostics.push(Diagnostic::error(code, message, span));
    }

    fn lower_statements(&mut self, statements: &[Stmt], module_level: bool) {
        for statement in statements {
            if self.current_terminated() {
                break;
            }
            self.lower_stmt(statement, module_level);
        }
    }

    fn lower_stmt(&mut self, stmt: &Stmt, module_level: bool) {
        self.set_source(stmt.span);
        match &stmt.kind {
            StmtKind::Variable { name, initializer } => {
                let value = initializer
                    .as_ref()
                    .map(|expr| self.lower_expr(expr))
                    .unwrap_or_else(|| self.builder.const_null());
                if module_level {
                    if let Some(slot) = self.context.module_layout.get(&name.value) {
                        self.builder.emit0(Inst::StoreModuleVar {
                            module: self.context.module,
                            slot,
                            value,
                        });
                    } else {
                        self.error("WREN-L010", "missing module slot for variable", name.span);
                    }
                } else if let Some(binding) = self
                    .context
                    .analysis
                    .declaration_map
                    .get(&name.span)
                    .copied()
                {
                    self.bind_initial(binding, value);
                }
            }
            StmtKind::Class(class) => self.lower_class_stmt(class, module_level),
            StmtKind::Import(import) => self.lower_import(import, module_level),
            StmtKind::Block(statements) => self.lower_statements(statements, false),
            StmtKind::If {
                condition,
                then_branch,
                else_branch,
            } => self.lower_if(condition, then_branch, else_branch.as_deref()),
            StmtKind::While { condition, body } => self.lower_while(condition, body),
            StmtKind::For {
                name,
                iterable,
                body,
            } => self.lower_for(name, iterable, body),
            StmtKind::Break => self.lower_break(stmt.span),
            StmtKind::Continue => self.lower_continue(stmt.span),
            StmtKind::Return(value) => self.lower_return(value.as_ref(), stmt.span),
            StmtKind::Expression(expr) => {
                let _ = self.lower_expr(expr);
            }
        }
    }

    fn lower_class_stmt(&mut self, class: &ClassDecl, module_level: bool) {
        let Some(plan) = self.context.class_plans.get(&class.name.span).cloned() else {
            self.error("WREN-L011", "class was not planned", class.name.span);
            return;
        };
        let superclass = class.superclass.as_ref().map(|expr| self.lower_expr(expr));
        let class_value = self.builder.emit1(
            Inst::CreateClass {
                class: plan.id,
                superclass,
            },
            Type::Class(ClassSet::singleton(plan.id)),
        );

        self.builder.emit0(Inst::StoreModuleVar {
            module: self.context.module,
            slot: plan.class_slot,
            value: class_value,
        });

        if module_level {
            if let Some(slot) = self.context.module_layout.get(&class.name.value) {
                self.builder.emit0(Inst::StoreModuleVar {
                    module: self.context.module,
                    slot,
                    value: class_value,
                });
            }
        } else if let Some(binding) = self
            .context
            .analysis
            .declaration_map
            .get(&class.name.span)
            .copied()
        {
            self.bind_initial(binding, class_value);
        }

        for method in &plan.methods {
            if let Some(function) = method.function {
                self.builder.emit0(Inst::BindMethod {
                    class: class_value,
                    method: method.symbol,
                    function,
                    is_static: method.bind_static,
                });
            } else if let Some(foreign) = method.foreign {
                self.builder.emit0(Inst::BindForeignMethod {
                    class: class_value,
                    method: method.symbol,
                    foreign,
                    is_static: method.bind_static,
                });
            }
        }
    }

    fn lower_import(&mut self, import: &ImportDecl, module_level: bool) {
        let module_name = match std::str::from_utf8(&import.module) {
            Ok(name) => name,
            Err(_) => {
                self.error(
                    "WREN-L012",
                    "module import names must be valid UTF-8 for the current WIR string table",
                    Span::empty(0),
                );
                return;
            }
        };
        let module_name = self
            .builder
            .emit1(
                Inst::ImportModule {
                    module_name: *self.context.string_ids.get(module_name).expect("import module name must be preinterned"),
                },
                Type::Module,
            );

        for binding in &import.bindings {
            let Some(name_id) = self.context.string_ids.get(&binding.imported.value).copied() else {
                self.error("WREN-L013", "imported name was not preinterned", binding.imported.span);
                continue;
            };
            let value = self.builder.emit1(
                Inst::ImportVariable {
                    module: module_name,
                    variable_name: name_id,
                },
                Type::Top,
            );
            if module_level {
                if let Some(slot) = self.context.module_layout.get(&binding.local.value) {
                    self.builder.emit0(Inst::StoreModuleVar {
                        module: self.context.module,
                        slot,
                        value,
                    });
                }
            } else if let Some(local) = self
                .context
                .analysis
                .declaration_map
                .get(&binding.local.span)
                .copied()
            {
                self.bind_initial(local, value);
            }
        }
    }

    fn lower_if(&mut self, condition: &Expr, then_stmt: &Stmt, else_stmt: Option<&Stmt>) {
        let condition = self.lower_expr(condition);
        let then_block = self.builder.new_block(&[]);
        let else_block = self.builder.new_block(&[]);
        self.builder
            .branch(condition, then_block, Vec::new(), else_block, Vec::new());

        let incoming = self.env.clone();
        let mut predecessors = Vec::new();

        self.builder.position_at(then_block);
        self.env = incoming.clone();
        self.lower_stmt(then_stmt, false);
        if !self.current_terminated() {
            predecessors.push((self.builder.current_block(), self.env.clone()));
        }

        self.builder.position_at(else_block);
        self.env = incoming.clone();
        if let Some(else_stmt) = else_stmt {
            self.lower_stmt(else_stmt, false);
        }
        if !self.current_terminated() {
            predecessors.push((self.builder.current_block(), self.env.clone()));
        }

        self.join_predecessors(predecessors, incoming);
    }

    fn join_predecessors(
        &mut self,
        predecessors: Vec<(ir::BlockId, HashMap<BindingId, Storage>)>,
        fallback: HashMap<BindingId, Storage>,
    ) {
        match predecessors.len() {
            0 => {
                self.env = fallback;
            }
            1 => {
                let (block, env) = predecessors.into_iter().next().unwrap();
                let join = self.builder.new_block(&[]);
                self.builder.position_at(block);
                self.builder.jump(join, Vec::new());
                self.builder.seal_block(join);
                self.builder.position_at(join);
                self.env = env;
            }
            _ => {
                let join = self.builder.new_block(&[]);
                let mut merged = fallback.clone();
                let mut params = Vec::new();

                let keys: Vec<_> = fallback.keys().copied().collect();
                for binding in keys {
                    let values: Vec<_> = predecessors
                        .iter()
                        .filter_map(|(_, env)| env.get(&binding).copied())
                        .collect();
                    if values.len() != predecessors.len() || values.is_empty() {
                        continue;
                    }
                    if values.iter().all(|value| *value == values[0]) {
                        merged.insert(binding, values[0]);
                        continue;
                    }
                    if values.iter().all(|value| matches!(value, Storage::Value(_))) {
                        let param = self.builder.add_block_param(join, Type::Top);
                        merged.insert(binding, Storage::Value(param));
                        params.push(binding);
                    }
                }

                for (block, env) in predecessors {
                    let args = params
                        .iter()
                        .map(|binding| match env[binding] {
                            Storage::Value(value) => value,
                            Storage::Cell(_) => unreachable!(),
                        })
                        .collect::<Vec<_>>();
                    self.builder.position_at(block);
                    self.builder.jump(join, args);
                }
                self.builder.seal_block(join);
                self.builder.position_at(join);
                self.env = merged;
            }
        }
    }

    fn lower_while(&mut self, condition: &Expr, body: &Stmt) {
        let carried = self.loop_carried_bindings(condition, body);
        let preheader_env = self.env.clone();

        let header = self.builder.new_block(&[]);
        let mut header_params = HashMap::new();
        for binding in &carried {
            let param = self.builder.add_block_param(header, Type::Top);
            header_params.insert(*binding, param);
        }
        let body_block = self.builder.new_block(&[]);
        let exit = self.builder.new_block(&[]);
        let mut exit_params = HashMap::new();
        for binding in &carried {
            let param = self.builder.add_block_param(exit, Type::Top);
            exit_params.insert(*binding, param);
        }

        let args = self.values_for_bindings(&carried, &preheader_env);
        self.builder.jump(header, args);
        self.builder.seal_block(header);
        self.builder.position_at(header);
        self.env = preheader_env.clone();
        for binding in &carried {
            self.env.insert(*binding, Storage::Value(header_params[binding]));
        }

        let cond = self.lower_expr(condition);
        let false_args = self.values_for_bindings(&carried, &self.env);
        self.builder
            .branch(cond, body_block, Vec::new(), exit, false_args);

        self.builder.position_at(body_block);
        let body_env = self.env.clone();
        self.env = body_env;
        self.loops.push(LoopContext {
            continue_target: header,
            break_target: exit,
            carried: carried.clone(),
            continue_prefix: Vec::new(),
        });
        self.lower_stmt(body, false);
        self.loops.pop();
        if !self.current_terminated() {
            let back_args = self.values_for_bindings(&carried, &self.env);
            self.builder.jump(header, back_args);
        }

        self.builder.seal_block(exit);
        self.builder.position_at(exit);
        self.env = preheader_env;
        for binding in &carried {
            self.env.insert(*binding, Storage::Value(exit_params[binding]));
        }
    }

    fn lower_for(&mut self, name: &Ident, iterable: &Expr, body: &Stmt) {
        let sequence = self.lower_expr(iterable);
        let carried = self.loop_carried_bindings(iterable, body);
        let base_env = self.env.clone();
        let iterator_initial = self.builder.const_null();

        let header = self.builder.new_block(&[]);
        let iterator_param = self.builder.add_block_param(header, Type::Top);
        let mut header_params = HashMap::new();
        for binding in &carried {
            header_params.insert(*binding, self.builder.add_block_param(header, Type::Top));
        }
        let body_block = self.builder.new_block(&[]);
        let exit = self.builder.new_block(&[]);
        let mut exit_params = HashMap::new();
        for binding in &carried {
            exit_params.insert(*binding, self.builder.add_block_param(exit, Type::Top));
        }

        let mut entry_args = vec![iterator_initial];
        entry_args.extend(self.values_for_bindings(&carried, &base_env));
        self.builder.jump(header, entry_args);
        self.builder.seal_block(header);
        self.builder.position_at(header);
        self.env = base_env.clone();
        for binding in &carried {
            self.env.insert(*binding, Storage::Value(header_params[binding]));
        }

        let iterate = self.context.symbols.expect("iterate", MethodKind::Method, 1);
        let iterator = self.builder.invoke(sequence, iterate, vec![iterator_param]);
        let false_args = self.values_for_bindings(&carried, &self.env);
        self.builder
            .branch(iterator, body_block, Vec::new(), exit, false_args);

        self.builder.position_at(body_block);
        let iterator_value_symbol = self
            .context
            .symbols
            .expect("iteratorValue", MethodKind::Method, 1);
        let item = self
            .builder
            .invoke(sequence, iterator_value_symbol, vec![iterator]);
        if let Some(binding) = self
            .context
            .analysis
            .declaration_map
            .get(&name.span)
            .copied()
        {
            self.bind_initial(binding, item);
        }

        self.loops.push(LoopContext {
            continue_target: header,
            break_target: exit,
            carried: carried.clone(),
            continue_prefix: vec![iterator],
        });
        self.lower_stmt(body, false);
        self.loops.pop();
        if !self.current_terminated() {
            let mut back = vec![iterator];
            back.extend(self.values_for_bindings(&carried, &self.env));
            self.builder.jump(header, back);
        }

        self.builder.seal_block(exit);
        self.builder.position_at(exit);
        self.env = base_env;
        for binding in &carried {
            self.env.insert(*binding, Storage::Value(exit_params[binding]));
        }
    }

    fn lower_break(&mut self, span: Span) {
        let Some(loop_context) = self.loops.last().cloned() else {
            self.error("WREN-L014", "'break' used outside a loop", span);
            return;
        };
        let args = self.values_for_bindings(&loop_context.carried, &self.env);
        self.builder.jump(loop_context.break_target, args);
    }

    fn lower_continue(&mut self, span: Span) {
        let Some(loop_context) = self.loops.last().cloned() else {
            self.error("WREN-L015", "'continue' used outside a loop", span);
            return;
        };
        let mut args = loop_context.continue_prefix;
        args.extend(self.values_for_bindings(&loop_context.carried, &self.env));
        self.builder.jump(loop_context.continue_target, args);
    }

    fn lower_return(&mut self, value: Option<&Expr>, span: Span) {
        if let Some(instance) = self.constructor_this {
            if value.is_some() {
                self.error(
                    "WREN-L016",
                    "constructors cannot return an explicit value",
                    span,
                );
            }
            self.builder.ret(instance);
            return;
        }
        let value = value
            .map(|expr| self.lower_expr(expr))
            .unwrap_or_else(|| self.builder.const_null());
        self.builder.ret(value);
    }

    fn loop_carried_bindings(&self, condition: &Expr, body: &Stmt) -> Vec<BindingId> {
        let mut writes = HashSet::new();
        collect_written_bindings_expr(condition, self.context.analysis, &mut writes);
        collect_written_bindings_stmt(body, self.context.analysis, &mut writes);
        let mut result: Vec<_> = writes
            .into_iter()
            .filter(|binding| matches!(self.env.get(binding), Some(Storage::Value(_))))
            .collect();
        result.sort_unstable();
        result
    }

    fn values_for_bindings(
        &self,
        bindings: &[BindingId],
        env: &HashMap<BindingId, Storage>,
    ) -> Vec<ValueId> {
        bindings
            .iter()
            .filter_map(|binding| match env.get(binding).copied() {
                Some(Storage::Value(value)) => Some(value),
                Some(Storage::Cell(_)) | None => None,
            })
            .collect()
    }

    fn lower_expr(&mut self, expr: &Expr) -> ValueId {
        self.set_source(expr.span);
        match &expr.kind {
            ExprKind::Null => self.builder.const_null(),
            ExprKind::Bool(value) => self.builder.const_bool(*value),
            ExprKind::Number(value) => self.builder.const_num(*value),
            ExprKind::String(bytes) => self.lower_string_literal(bytes, expr.span),
            ExprKind::InterpolatedString(parts) => self.lower_interpolated_string(parts, expr.span),
            ExprKind::Name(name) => self.lower_name(name),
            ExprKind::Field(name) => self.lower_field(name),
            ExprKind::StaticField(name) => self.lower_static_field(name),
            ExprKind::This => self.lower_this(expr.span),
            ExprKind::Super(super_expr) => self.lower_super(super_expr, expr.span),
            ExprKind::List(items) => self.lower_list(items),
            ExprKind::Map(entries) => self.lower_map(entries),
            ExprKind::Grouped(inner) => self.lower_expr(inner),
            ExprKind::Unary { operator, operand } => self.lower_unary(*operator, operand),
            ExprKind::Binary {
                left,
                operator,
                right,
            } => self.lower_binary(left, *operator, right),
            ExprKind::Logical {
                left,
                operator,
                right,
            } => self.lower_logical(left, *operator, right),
            ExprKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => self.lower_conditional(condition, then_expr, else_expr),
            ExprKind::Assign { target, value } => self.lower_assignment(target, value),
            ExprKind::Invoke(invoke) => self.lower_invoke(invoke, expr.span),
            ExprKind::Subscript { receiver, args } => self.lower_subscript(receiver, args),
            ExprKind::Await(promise) => {
                let promise = self.lower_expr(promise);
                if !self.context.allows_await {
                    self.error(
                        "WREN-L040",
                        "`await` is only valid inside an async function or at module scope",
                        expr.span,
                    );
                }
                self.builder.emit1(Inst::Await { promise }, Type::Top)
            }
            ExprKind::Closure(body) | ExprKind::AsyncClosure(body) => self.lower_closure(body),
        }
    }

    fn lower_string_literal(&mut self, bytes: &[u8], span: Span) -> ValueId {
        match std::str::from_utf8(bytes) {
            Ok(text) => self.builder.const_string(text),
            Err(_) => {
                self.error(
                    "WREN-L017",
                    "non-UTF-8 Wren string literals are not representable by the current WIR StringId table",
                    span,
                );
                self.builder.const_string(String::from_utf8_lossy(bytes))
            }
        }
    }

    fn lower_interpolated_string(&mut self, parts: &[StringPart], span: Span) -> ValueId {
        let mut result = self.builder.const_string("");
        let to_string = self.context.symbols.expect("toString", MethodKind::Getter, 0);
        for part in parts {
            let piece = match part {
                StringPart::Bytes(bytes) => self.lower_string_literal(bytes, span),
                StringPart::Expression(expr) => {
                    let value = self.lower_expr(expr);
                    let site = self.builder.fresh_site();
                    self.builder.emit1(
                        Inst::Invoke {
                            site,
                            receiver: value,
                            method: to_string,
                            args: Vec::new(),
                        },
                        Type::String,
                    )
                }
            };
            result = self.builder.emit1(
                Inst::StringConcat {
                    left: result,
                    right: piece,
                },
                Type::String,
            );
        }
        result
    }

    fn lower_name(&mut self, name: &Ident) -> ValueId {
        if let Some(binding) = self.context.analysis.use_map.get(&name.span).copied() {
            return self.read_binding(binding);
        }
        if self.context.method_context.is_some() && is_implicit_receiver_name(&name.value) {
            let receiver = self.lower_this(name.span);
            let method = self.context.symbols.expect(&name.value, MethodKind::Getter, 0);
            return self.builder.invoke(receiver, method, Vec::new());
        }
        if let Some(slot) = self.context.module_layout.get(&name.value) {
            return self.builder.emit1(
                Inst::LoadModuleVar {
                    module: self.context.module,
                    slot,
                },
                Type::Top,
            );
        }
        self.error(
            "WREN-L018",
            format!("undefined variable '{}'", name.value),
            name.span,
        );
        self.builder.const_null()
    }

    fn lower_this(&mut self, span: Span) -> ValueId {
        let node = &self.context.analysis.nodes[self.context.node_id.0];
        let binding = node.this_binding.or_else(|| {
            node.captures.iter().copied().find(|binding| {
                self.context.analysis.binding(*binding).synthetic_this
            })
        });
        if let Some(binding) = binding {
            return self.read_binding(binding);
        }
        self.error("WREN-L019", "'this' used outside a method", span);
        self.builder.const_null()
    }

    fn lower_field(&mut self, name: &Ident) -> ValueId {
        let Some(method) = self.context.method_context.clone() else {
            self.error("WREN-L020", "instance field used outside a class method", name.span);
            return self.builder.const_null();
        };
        if method.static_method {
            self.error("WREN-L039", "instance fields cannot be used in static methods", name.span);
            return self.builder.const_null();
        }
        let Some(slot) = method.fields.get(&name.value).copied() else {
            self.error(
                "WREN-L021",
                format!("unknown instance field '{}'", name.value),
                name.span,
            );
            return self.builder.const_null();
        };
        let this = self.lower_this(name.span);
        self.builder.load_field(this, slot, Type::Top)
    }

    fn lower_static_field(&mut self, name: &Ident) -> ValueId {
        let Some(method) = self.context.method_context.clone() else {
            self.error("WREN-L022", "static field used outside a class method", name.span);
            return self.builder.const_null();
        };
        let Some(slot) = method.static_fields.get(&name.value).copied() else {
            self.error(
                "WREN-L023",
                format!("unknown static field '{}'", name.value),
                name.span,
            );
            return self.builder.const_null();
        };
        let class = self.lower_lexical_class(method.class);
        self.builder
            .emit1(Inst::LoadStaticField { class, slot }, Type::Top)
    }

    fn lower_lexical_class(&mut self, class: ClassId) -> ValueId {
        let plan = self.context.class_plans.values().find(|plan| plan.id == class).unwrap();
        self.builder.emit1(
            Inst::LoadModuleVar { module: self.context.module, slot: plan.class_slot },
            Type::Class(ClassSet::singleton(class)),
        )
    }

    fn lower_super(&mut self, super_expr: &front::SuperExpr, span: Span) -> ValueId {
        let Some(method) = self.context.method_context.clone() else {
            self.error("WREN-L025", "'super' used outside a method", span);
            return self.builder.const_null();
        };
        let receiver = self.lower_this(span);
        let (name, kind, arity) = if let Some(name) = &super_expr.method {
            (
                name.value.clone(),
                match super_expr.syntax {
                    InvokeSyntax::Getter => MethodKind::Getter,
                    InvokeSyntax::Method => MethodKind::Method,
                },
                super_expr.args.len(),
            )
        } else if let Some(symbol) = method.method_symbol {
            let Some(signature) = self.context_symbol_signature(symbol) else {
                self.error("WREN-L026", "cannot resolve enclosing method signature", span);
                return self.builder.const_null();
            };
            (signature.0, signature.1, super_expr.args.len())
        } else {
            self.error("WREN-L027", "unnamed super call has no enclosing method", span);
            return self.builder.const_null();
        };
        // A missing superclass implementation is a runtime dispatch error, not
        // a missing-symbol panic in the compiler (including changed arities).
        let symbol = self.builder.intern_method(&name, kind, arity as u8);
        let args = super_expr.args.iter().map(|arg| self.lower_expr(arg)).collect();
        self.builder.invoke_super(method.class, receiver, symbol, args)
    }

    fn context_symbol_signature(&self, symbol: MethodSymbol) -> Option<(String, MethodKind)> {
        self.context
            .symbols
            .methods
            .iter()
            .find_map(|(key, value)| (*value == symbol).then(|| (key.name.clone(), key.kind)))
    }

    fn lower_list(&mut self, items: &[Expr]) -> ValueId {
        let capacity = self.builder.const_i32(items.len().min(i32::MAX as usize) as i32);
        let list = self.builder.emit1(
            Inst::AllocList {
                capacity: Some(capacity),
            },
            Type::List(None),
        );
        for item in items {
            let value = self.lower_expr(item);
            self.builder.emit0(Inst::ListAppend { list, value });
        }
        list
    }

    fn lower_map(&mut self, entries: &[front::MapEntry]) -> ValueId {
        let map = self.builder.emit1(Inst::AllocMap, Type::Map);
        for entry in entries {
            let key = self.lower_expr(&entry.key);
            let value = self.lower_expr(&entry.value);
            self.builder.emit0(Inst::MapSet { map, key, value });
        }
        map
    }

    fn lower_unary(&mut self, operator: UnaryOp, operand: &Expr) -> ValueId {
        let receiver = self.lower_expr(operand);
        let name = match operator {
            UnaryOp::Negate => "-",
            UnaryOp::Not => "!",
            UnaryOp::Complement => "~",
        };
        let method = self.context.symbols.expect(name, MethodKind::Method, 0);
        self.builder.invoke(receiver, method, Vec::new())
    }

    fn lower_binary(&mut self, left: &Expr, operator: BinaryOp, right: &Expr) -> ValueId {
        let left = self.lower_expr(left);
        let right = self.lower_expr(right);
        if operator == BinaryOp::Is {
            return self
                .builder
                .emit1(Inst::IsInstance { value: left, class: right }, Type::Bool);
        }
        let name = binary_operator_name(operator);
        let method = self.context.symbols.expect(name, MethodKind::Method, 1);
        self.builder.invoke(left, method, vec![right])
    }

    fn lower_logical(&mut self, left: &Expr, operator: LogicalOp, right: &Expr) -> ValueId {
        let left_value = self.lower_expr(left);
        let rhs_block = self.builder.new_block(&[]);
        let short_block = self.builder.new_block(&[]);
        let join = self.builder.new_block(&[]);
        let result_param = self.builder.add_block_param(join, Type::Top);
        let incoming = self.env.clone();

        match operator {
            LogicalOp::And => self.builder.branch(
                left_value,
                rhs_block,
                Vec::new(),
                short_block,
                Vec::new(),
            ),
            LogicalOp::Or => self.builder.branch(
                left_value,
                short_block,
                Vec::new(),
                rhs_block,
                Vec::new(),
            ),
        }

        self.builder.position_at(short_block);
        self.env = incoming.clone();
        let short_end = self.builder.current_block();
        let short_env = self.env.clone();

        self.builder.position_at(rhs_block);
        self.env = incoming.clone();
        let rhs_value = self.lower_expr(right);
        let rhs_end = self.builder.current_block();
        let rhs_env = self.env.clone();

        let mut merged_env = incoming.clone();
        let mut env_params = Vec::new();
        for binding in incoming.keys().copied().collect::<Vec<_>>() {
            match (short_env.get(&binding).copied(), rhs_env.get(&binding).copied()) {
                (Some(Storage::Value(a)), Some(Storage::Value(b))) if a != b => {
                    let param = self.builder.add_block_param(join, Type::Top);
                    merged_env.insert(binding, Storage::Value(param));
                    env_params.push(binding);
                }
                (Some(value), Some(_)) => {
                    merged_env.insert(binding, value);
                }
                _ => {}
            }
        }

        self.builder.position_at(short_end);
        let mut short_args = vec![left_value];
        short_args.extend(env_params.iter().map(|binding| match short_env[binding] {
            Storage::Value(value) => value,
            Storage::Cell(_) => unreachable!(),
        }));
        self.builder.jump(join, short_args);

        self.builder.position_at(rhs_end);
        let mut rhs_args = vec![rhs_value];
        rhs_args.extend(env_params.iter().map(|binding| match rhs_env[binding] {
            Storage::Value(value) => value,
            Storage::Cell(_) => unreachable!(),
        }));
        self.builder.jump(join, rhs_args);

        self.builder.seal_block(join);
        self.builder.position_at(join);
        self.env = merged_env;
        result_param
    }

    fn lower_conditional(&mut self, condition: &Expr, then_expr: &Expr, else_expr: &Expr) -> ValueId {
        let condition = self.lower_expr(condition);
        let then_block = self.builder.new_block(&[]);
        let else_block = self.builder.new_block(&[]);
        let join = self.builder.new_block(&[]);
        let result_param = self.builder.add_block_param(join, Type::Top);
        self.builder
            .branch(condition, then_block, Vec::new(), else_block, Vec::new());
        let incoming = self.env.clone();

        self.builder.position_at(then_block);
        self.env = incoming.clone();
        let then_value = self.lower_expr(then_expr);
        let then_end = self.builder.current_block();
        let then_env = self.env.clone();

        self.builder.position_at(else_block);
        self.env = incoming.clone();
        let else_value = self.lower_expr(else_expr);
        let else_end = self.builder.current_block();
        let else_env = self.env.clone();

        let mut merged_env = incoming.clone();
        let mut env_params = Vec::new();
        for binding in incoming.keys().copied().collect::<Vec<_>>() {
            match (then_env.get(&binding).copied(), else_env.get(&binding).copied()) {
                (Some(Storage::Value(a)), Some(Storage::Value(b))) if a != b => {
                    let param = self.builder.add_block_param(join, Type::Top);
                    merged_env.insert(binding, Storage::Value(param));
                    env_params.push(binding);
                }
                (Some(value), Some(_)) => {
                    merged_env.insert(binding, value);
                }
                _ => {}
            }
        }

        self.builder.position_at(then_end);
        let mut then_args = vec![then_value];
        then_args.extend(env_params.iter().map(|binding| match then_env[binding] {
            Storage::Value(value) => value,
            Storage::Cell(_) => unreachable!(),
        }));
        self.builder.jump(join, then_args);

        self.builder.position_at(else_end);
        let mut else_args = vec![else_value];
        else_args.extend(env_params.iter().map(|binding| match else_env[binding] {
            Storage::Value(value) => value,
            Storage::Cell(_) => unreachable!(),
        }));
        self.builder.jump(join, else_args);

        self.builder.seal_block(join);
        self.builder.position_at(join);
        self.env = merged_env;
        result_param
    }

    fn lower_assignment(&mut self, target: &Expr, value: &Expr) -> ValueId {
        let value = self.lower_expr(value);
        match &target.kind {
            ExprKind::Name(name) => {
                if let Some(binding) = self.context.analysis.use_map.get(&name.span).copied() {
                    self.write_binding(binding, value);
                } else if self.context.method_context.is_some() && is_implicit_receiver_name(&name.value) {
                    let receiver = self.lower_this(name.span);
                    let setter = self.context.symbols.expect(&name.value, MethodKind::Setter, 1);
                    let _ = self.builder.invoke(receiver, setter, vec![value]);
                } else if let Some(slot) = self.context.module_layout.get(&name.value) {
                    if BuiltinId::ALL.iter().any(|builtin| builtin.name() == name.value) {
                        self.error(
                            "WREN-L038",
                            format!("cannot assign to core global '{}'", name.value),
                            name.span,
                        );
                    } else {
                        self.builder.emit0(Inst::StoreModuleVar {
                            module: self.context.module,
                            slot,
                            value,
                        });
                    }
                } else {
                    self.error(
                        "WREN-L028",
                        format!("assignment to undefined variable '{}'", name.value),
                        name.span,
                    );
                }
            }
            ExprKind::Field(name) => {
                let Some(method) = self.context.method_context.clone() else {
                    self.error("WREN-L029", "field assignment outside a method", name.span);
                    return value;
                };
                if method.static_method {
                    self.error("WREN-L039", "instance fields cannot be used in static methods", name.span);
                    return value;
                }
                if let Some(slot) = method.fields.get(&name.value).copied() {
                    let this = self.lower_this(name.span);
                    self.builder.store_field(this, slot, value);
                } else {
                    self.error("WREN-L030", "unknown instance field", name.span);
                }
            }
            ExprKind::StaticField(name) => {
                let Some(method) = self.context.method_context.clone() else {
                    self.error("WREN-L031", "static field assignment outside a method", name.span);
                    return value;
                };
                if let Some(slot) = method.static_fields.get(&name.value).copied() {
                    let class = self.lower_lexical_class(method.class);
                    self.builder.emit0(Inst::StoreStaticField { class, slot, value });
                } else {
                    self.error("WREN-L033", "unknown static field", name.span);
                }
            }
            ExprKind::Invoke(invoke) => {
                let receiver = invoke
                    .receiver
                    .as_ref()
                    .map(|receiver| self.lower_expr(receiver))
                    .unwrap_or_else(|| self.lower_this(target.span));
                let setter = self
                    .context
                    .symbols
                    .expect(&invoke.name.value, MethodKind::Setter, 1);
                let _ = self.builder.invoke(receiver, setter, vec![value]);
            }
            ExprKind::Subscript { receiver, args } => {
                let receiver = self.lower_expr(receiver);
                let mut lowered: Vec<_> = args.iter().map(|arg| self.lower_expr(arg)).collect();
                lowered.push(value);
                let method = self.context.symbols.expect(
                    "[]=",
                    MethodKind::SubscriptSetter,
                    lowered.len(),
                );
                let _ = self.builder.invoke(receiver, method, lowered);
            }
            _ => {
                self.error("WREN-L034", "invalid assignment target", target.span);
            }
        }
        value
    }

    fn lower_invoke(&mut self, invoke: &front::InvokeExpr, span: Span) -> ValueId {
        let receiver = invoke
            .receiver
            .as_ref()
            .map(|receiver| self.lower_expr(receiver))
            .unwrap_or_else(|| self.lower_this(span));
        let args: Vec<_> = invoke.args.iter().map(|arg| self.lower_expr(arg)).collect();
        let kind = match invoke.syntax {
            InvokeSyntax::Getter => MethodKind::Getter,
            InvokeSyntax::Method => MethodKind::Method,
        };
        let symbol = self
            .context
            .symbols
            .expect(&invoke.name.value, kind, args.len());
        self.builder.invoke(receiver, symbol, args)
    }

    fn lower_subscript(&mut self, receiver: &Expr, args: &[Expr]) -> ValueId {
        let receiver = self.lower_expr(receiver);
        let args: Vec<_> = args.iter().map(|arg| self.lower_expr(arg)).collect();
        let symbol = self
            .context
            .symbols
            .expect("[]", MethodKind::SubscriptGetter, args.len());
        self.builder.invoke(receiver, symbol, args)
    }

    fn lower_closure(&mut self, body: &FunctionBody) -> ValueId {
        let Some(node_id) = self.context.analysis.closure_map.get(&body.span).copied() else {
            self.error("WREN-L035", "closure was not discovered during capture analysis", body.span);
            return self.builder.const_null();
        };
        let Some(function) = self.context.closure_ids.get(&node_id).copied() else {
            self.error("WREN-L036", "closure function was not compiled", body.span);
            return self.builder.const_null();
        };
        let captures = self.context.analysis.nodes[node_id.0]
            .captures
            .iter()
            .map(|binding| self.capture_operand(*binding))
            .collect();
        self.builder.make_closure(function, captures)
    }

    fn read_binding(&mut self, binding: BindingId) -> ValueId {
        match self.env.get(&binding).copied() {
            Some(Storage::Value(value)) => value,
            Some(Storage::Cell(cell)) => self.builder.load_cell(cell, Type::Top),
            None => {
                let info = self.context.analysis.binding(binding);
                self.error(
                    "WREN-L037",
                    format!("internal lowering error: binding '{}' is unavailable", info.name),
                    info.declaration,
                );
                self.builder.const_null()
            }
        }
    }

    fn write_binding(&mut self, binding: BindingId, value: ValueId) {
        match self.env.get(&binding).copied() {
            Some(Storage::Cell(cell)) => self.builder.store_cell(cell, value),
            Some(Storage::Value(_)) => {
                self.env.insert(binding, Storage::Value(value));
            }
            None => {
                let info = self.context.analysis.binding(binding);
                self.error(
                    "WREN-L038",
                    format!("internal lowering error: binding '{}' is unavailable", info.name),
                    info.declaration,
                );
            }
        }
    }

    fn capture_operand(&mut self, binding: BindingId) -> ValueId {
        match self.env.get(&binding).copied() {
            Some(Storage::Cell(cell)) => cell,
            Some(Storage::Value(value)) => value,
            None => self.read_binding(binding),
        }
    }
}

// =============================================================================
// Collection / planning helpers
// =============================================================================

fn is_implicit_receiver_name(name: &str) -> bool {
    name.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
}

fn function_node_span(node: &FunctionNode) -> Span {
    match &node.ast {
        FunctionAst::Statements(statements) => statements
            .first()
            .zip(statements.last())
            .map(|(first, last)| first.span.join(last.span))
            .unwrap_or_default(),
        FunctionAst::Body(body) => body.span,
    }
}

fn source_span(source: &str, file: ir::StringId, span: Span) -> SourceSpan {
    let start = span.start.min(source.len());
    let prefix = &source[..start];
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() as u32 + 1;
    let line_start = prefix.rfind('\n').map(|index| index + 1).unwrap_or(0);
    let column = source[line_start..start].chars().count() as u32 + 1;
    SourceSpan {
        file: Some(file),
        line,
        column,
        length: span.len().min(u32::MAX as usize) as u32,
    }
}

fn method_formal_params(signature: &MethodSignatureAst) -> Vec<Ident> {
    match signature {
        MethodSignatureAst::Getter { .. } | MethodSignatureAst::PrefixOperator { .. } => Vec::new(),
        MethodSignatureAst::Method { params, .. }
        | MethodSignatureAst::SubscriptGetter { params } => params.clone(),
        MethodSignatureAst::Setter { param, .. }
        | MethodSignatureAst::InfixOperator { param, .. } => vec![param.clone()],
        MethodSignatureAst::SubscriptSetter { params, value } => {
            let mut result = params.clone();
            result.push(value.clone());
            result
        }
    }
}

fn method_signature_key(signature: &MethodSignatureAst) -> Option<(String, MethodKind, usize)> {
    Some(match signature {
        MethodSignatureAst::Getter { name } => (name.value.clone(), MethodKind::Getter, 0),
        MethodSignatureAst::Method { name, params } => {
            (name.value.clone(), MethodKind::Method, params.len())
        }
        MethodSignatureAst::Setter { name, .. } => (name.value.clone(), MethodKind::Setter, 1),
        MethodSignatureAst::PrefixOperator { operator } => {
            (operator.as_str().to_owned(), MethodKind::Method, 0)
        }
        MethodSignatureAst::InfixOperator { operator, .. } => {
            (operator.as_str().to_owned(), MethodKind::Method, 1)
        }
        MethodSignatureAst::SubscriptGetter { params } => {
            ("[]".to_owned(), MethodKind::SubscriptGetter, params.len())
        }
        MethodSignatureAst::SubscriptSetter { params, .. } => (
            "[]=".to_owned(),
            MethodKind::SubscriptSetter,
            params.len() + 1,
        ),
    })
}

fn display_method_signature(signature: &MethodSignatureAst) -> String {
    let (name, _, arity) = method_signature_key(signature).unwrap();
    if arity == 0 {
        name
    } else {
        format!("{}({})", name, (0..arity).map(|_| "_").collect::<Vec<_>>().join(","))
    }
}

fn binary_operator_name(operator: BinaryOp) -> &'static str {
    match operator {
        BinaryOp::Multiply => "*",
        BinaryOp::Divide => "/",
        BinaryOp::Modulo => "%",
        BinaryOp::Add => "+",
        BinaryOp::Subtract => "-",
        BinaryOp::InclusiveRange => "..",
        BinaryOp::ExclusiveRange => "...",
        BinaryOp::ShiftLeft => "<<",
        BinaryOp::ShiftRight => ">>",
        BinaryOp::BitAnd => "&",
        BinaryOp::BitXor => "^",
        BinaryOp::BitOr => "|",
        BinaryOp::Less => "<",
        BinaryOp::LessEqual => "<=",
        BinaryOp::Greater => ">",
        BinaryOp::GreaterEqual => ">=",
        BinaryOp::Is => "is",
        BinaryOp::Equal => "==",
        BinaryOp::NotEqual => "!=",
    }
}

fn collect_class_declarations<'a>(statements: &'a [Stmt], out: &mut Vec<(&'a ClassDecl, bool)>) {
    fn walk<'a>(statements: &'a [Stmt], out: &mut Vec<(&'a ClassDecl, bool)>, top: bool) {
        for statement in statements {
            match &statement.kind {
                StmtKind::Class(class) => {
                    out.push((class, top));
                    for method in &class.methods {
                        if let Some(body) = &method.body {
                            if let FunctionBodyKind::Statements(statements) = &body.kind {
                                walk(statements, out, false);
                            }
                        }
                    }
                }
                StmtKind::Block(statements) => walk(statements, out, false),
                StmtKind::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    walk(std::slice::from_ref(then_branch.as_ref()), out, false);
                    if let Some(else_branch) = else_branch {
                        walk(std::slice::from_ref(else_branch.as_ref()), out, false);
                    }
                }
                StmtKind::While { body, .. } | StmtKind::For { body, .. } => {
                    walk(std::slice::from_ref(body.as_ref()), out, false)
                }
                _ => {}
            }
        }
    }
    walk(statements, out, true)
}

fn find_class_by_id<'a>(
    program: &ir::Program,
    id: ClassId,
    module: &'a ModuleAst,
) -> Option<(&'a ClassDecl, bool)> {
    let info = program.class(id)?;
    let name = program.string(info.name)?;
    let mut declarations = Vec::new();
    collect_class_declarations(&module.statements, &mut declarations);
    declarations
        .into_iter()
        .find(|(class, top)| *top && class.name.value == name)
}

fn collect_fields_in_body<'a>(body: &'a FunctionBody, instance: &mut SmallVec<[&'a str; 8]>, statics: &mut SmallVec<[&'a str; 8]>) {
    match &body.kind {
        FunctionBodyKind::Expression(expr) => collect_fields_expr(expr, instance, statics),
        FunctionBodyKind::Statements(statements) => {
            for statement in statements {
                collect_fields_stmt(statement, instance, statics);
            }
        }
    }
}

fn collect_fields_stmt<'a>(stmt: &'a Stmt, instance: &mut SmallVec<[&'a str; 8]>, statics: &mut SmallVec<[&'a str; 8]>) {
    match &stmt.kind {
        StmtKind::Variable { initializer, .. } => {
            if let Some(expr) = initializer {
                collect_fields_expr(expr, instance, statics)
            }
        }
        StmtKind::Class(_) | StmtKind::Import(_) | StmtKind::Break | StmtKind::Continue => {}
        StmtKind::Block(statements) => {
            for statement in statements {
                collect_fields_stmt(statement, instance, statics)
            }
        }
        StmtKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_fields_expr(condition, instance, statics);
            collect_fields_stmt(then_branch, instance, statics);
            if let Some(else_branch) = else_branch {
                collect_fields_stmt(else_branch, instance, statics)
            }
        }
        StmtKind::While { condition, body } => {
            collect_fields_expr(condition, instance, statics);
            collect_fields_stmt(body, instance, statics)
        }
        StmtKind::For { iterable, body, .. } => {
            collect_fields_expr(iterable, instance, statics);
            collect_fields_stmt(body, instance, statics)
        }
        StmtKind::Return(value) => {
            if let Some(value) = value {
                collect_fields_expr(value, instance, statics)
            }
        }
        StmtKind::Expression(expr) => collect_fields_expr(expr, instance, statics),
    }
}

fn collect_fields_expr<'a>(expr: &'a Expr, instance: &mut SmallVec<[&'a str; 8]>, statics: &mut SmallVec<[&'a str; 8]>) {
    match &expr.kind {
        ExprKind::Field(name) => instance.push(name.value.as_str()),
        ExprKind::StaticField(name) => statics.push(name.value.as_str()),
        ExprKind::InterpolatedString(parts) => {
            for part in parts {
                if let StringPart::Expression(expr) = part {
                    collect_fields_expr(expr, instance, statics)
                }
            }
        }
        ExprKind::List(items) => {
            for item in items {
                collect_fields_expr(item, instance, statics)
            }
        }
        ExprKind::Map(entries) => {
            for entry in entries {
                collect_fields_expr(&entry.key, instance, statics);
                collect_fields_expr(&entry.value, instance, statics);
            }
        }
        ExprKind::Grouped(inner) => collect_fields_expr(inner, instance, statics),
        ExprKind::Unary { operand, .. } => collect_fields_expr(operand, instance, statics),
        ExprKind::Binary { left, right, .. } | ExprKind::Logical { left, right, .. } => {
            collect_fields_expr(left, instance, statics);
            collect_fields_expr(right, instance, statics);
        }
        ExprKind::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            collect_fields_expr(condition, instance, statics);
            collect_fields_expr(then_expr, instance, statics);
            collect_fields_expr(else_expr, instance, statics);
        }
        ExprKind::Assign { target, value } => {
            collect_fields_expr(target, instance, statics);
            collect_fields_expr(value, instance, statics);
        }
        ExprKind::Invoke(invoke) => {
            if let Some(receiver) = &invoke.receiver {
                collect_fields_expr(receiver, instance, statics)
            }
            for arg in &invoke.args {
                collect_fields_expr(arg, instance, statics)
            }
        }
        ExprKind::Subscript { receiver, args } => {
            collect_fields_expr(receiver, instance, statics);
            for arg in args {
                collect_fields_expr(arg, instance, statics)
            }
        }
        ExprKind::Await(value) => collect_fields_expr(value, instance, statics),
        ExprKind::Closure(body) | ExprKind::AsyncClosure(body) => collect_fields_in_body(body, instance, statics),
        ExprKind::Super(super_expr) => {
            for arg in &super_expr.args {
                collect_fields_expr(arg, instance, statics)
            }
        }
        ExprKind::Null
        | ExprKind::Bool(_)
        | ExprKind::Number(_)
        | ExprKind::String(_)
        | ExprKind::Name(_)
        | ExprKind::This => {}
    }
}

fn dedup_preserve_order(values: &mut SmallVec<[&str; 8]>) {
    let mut seen = HashSet::new();
    values.retain(|value| seen.insert(*value));
}

fn collect_method_keys_in_statements(statements: &[Stmt], out: &mut HashSet<MethodKey>) {
    for statement in statements {
        collect_method_keys_stmt(statement, out)
    }
}

fn collect_method_keys_stmt(stmt: &Stmt, out: &mut HashSet<MethodKey>) {
    match &stmt.kind {
        StmtKind::Variable { initializer, .. } => {
            if let Some(expr) = initializer {
                collect_method_keys_expr(expr, out)
            }
        }
        StmtKind::Class(class) => {
            if let Some(superclass) = &class.superclass {
                collect_method_keys_expr(superclass, out)
            }
            for method in &class.methods {
                if let Some((name, kind, arity)) = method_signature_key(&method.signature) {
                    out.insert(MethodKey {
                        name,
                        kind,
                        arity: arity as u8,
                    });
                }
                // Setter symbols may be needed by assignment syntax even when
                // only a getter-shaped AST invocation exists.
                if let Some(body) = &method.body {
                    collect_method_keys_body(body, out)
                }
            }
        }
        StmtKind::Import(_) | StmtKind::Break | StmtKind::Continue => {}
        StmtKind::Block(statements) => collect_method_keys_in_statements(statements, out),
        StmtKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_method_keys_expr(condition, out);
            collect_method_keys_stmt(then_branch, out);
            if let Some(else_branch) = else_branch {
                collect_method_keys_stmt(else_branch, out)
            }
        }
        StmtKind::While { condition, body } => {
            collect_method_keys_expr(condition, out);
            collect_method_keys_stmt(body, out)
        }
        StmtKind::For { iterable, body, .. } => {
            collect_method_keys_expr(iterable, out);
            collect_method_keys_stmt(body, out)
        }
        StmtKind::Return(value) => {
            if let Some(value) = value {
                collect_method_keys_expr(value, out)
            }
        }
        StmtKind::Expression(expr) => collect_method_keys_expr(expr, out),
    }
}

fn collect_method_keys_body(body: &FunctionBody, out: &mut HashSet<MethodKey>) {
    match &body.kind {
        FunctionBodyKind::Expression(expr) => collect_method_keys_expr(expr, out),
        FunctionBodyKind::Statements(statements) => collect_method_keys_in_statements(statements, out),
    }
}

fn collect_method_keys_expr(expr: &Expr, out: &mut HashSet<MethodKey>) {
    match &expr.kind {
        ExprKind::Unary { operator, operand } => {
            let name = match operator {
                UnaryOp::Negate => "-",
                UnaryOp::Not => "!",
                UnaryOp::Complement => "~",
            };
            out.insert(MethodKey {
                name: name.to_owned(),
                kind: MethodKind::Method,
                arity: 0,
            });
            collect_method_keys_expr(operand, out)
        }
        ExprKind::Binary { left, operator, right } => {
            if *operator != BinaryOp::Is {
                out.insert(MethodKey {
                    name: binary_operator_name(*operator).to_owned(),
                    kind: MethodKind::Method,
                    arity: 1,
                });
            }
            collect_method_keys_expr(left, out);
            collect_method_keys_expr(right, out);
        }
        ExprKind::Invoke(invoke) => {
            let kind = match invoke.syntax {
                InvokeSyntax::Getter => MethodKind::Getter,
                InvokeSyntax::Method => MethodKind::Method,
            };
            out.insert(MethodKey {
                name: invoke.name.value.clone(),
                kind,
                arity: invoke.args.len() as u8,
            });
            out.insert(MethodKey {
                name: invoke.name.value.clone(),
                kind: MethodKind::Setter,
                arity: 1,
            });
            if let Some(receiver) = &invoke.receiver {
                collect_method_keys_expr(receiver, out)
            }
            for arg in &invoke.args {
                collect_method_keys_expr(arg, out)
            }
        }
        ExprKind::Subscript { receiver, args } => {
            out.insert(MethodKey {
                name: "[]".to_owned(),
                kind: MethodKind::SubscriptGetter,
                arity: args.len() as u8,
            });
            // A maximum-arity getter is valid even though a corresponding
            // setter would exceed the argument limit once its value is added.
            if args.len() < MAX_WREN_ARITY {
                out.insert(MethodKey {
                    name: "[]=".to_owned(),
                    kind: MethodKind::SubscriptSetter,
                    arity: (args.len() + 1) as u8,
                });
            }
            collect_method_keys_expr(receiver, out);
            for arg in args {
                collect_method_keys_expr(arg, out)
            }
        }
        ExprKind::Assign { target, value } => {
            if let ExprKind::Name(name) = &target.kind {
                out.insert(MethodKey {
                    name: name.value.clone(),
                    kind: MethodKind::Setter,
                    arity: 1,
                });
            }
            collect_method_keys_expr(target, out);
            collect_method_keys_expr(value, out)
        }
        ExprKind::Super(super_expr) => {
            if let Some(method) = &super_expr.method {
                out.insert(MethodKey {
                    name: method.value.clone(),
                    kind: match super_expr.syntax {
                        InvokeSyntax::Getter => MethodKind::Getter,
                        InvokeSyntax::Method => MethodKind::Method,
                    },
                    arity: super_expr.args.len() as u8,
                });
            }
            for arg in &super_expr.args {
                collect_method_keys_expr(arg, out)
            }
        }
        ExprKind::InterpolatedString(parts) => {
            for part in parts {
                if let StringPart::Expression(expr) = part {
                    collect_method_keys_expr(expr, out)
                }
            }
        }
        ExprKind::List(items) => {
            for item in items {
                collect_method_keys_expr(item, out)
            }
        }
        ExprKind::Map(entries) => {
            for entry in entries {
                collect_method_keys_expr(&entry.key, out);
                collect_method_keys_expr(&entry.value, out);
            }
        }
        ExprKind::Grouped(inner) => collect_method_keys_expr(inner, out),
        ExprKind::Logical { left, right, .. } => {
            collect_method_keys_expr(left, out);
            collect_method_keys_expr(right, out)
        }
        ExprKind::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            collect_method_keys_expr(condition, out);
            collect_method_keys_expr(then_expr, out);
            collect_method_keys_expr(else_expr, out)
        }
        ExprKind::Await(value) => collect_method_keys_expr(value, out),
        ExprKind::Closure(body) | ExprKind::AsyncClosure(body) => collect_method_keys_body(body, out),
        ExprKind::Name(name) => {
            out.insert(MethodKey {
                name: name.value.clone(),
                kind: MethodKind::Getter,
                arity: 0,
            });
            out.insert(MethodKey {
                name: name.value.clone(),
                kind: MethodKind::Setter,
                arity: 1,
            });
        }
        ExprKind::Null
        | ExprKind::Bool(_)
        | ExprKind::Number(_)
        | ExprKind::String(_)
        | ExprKind::Field(_)
        | ExprKind::StaticField(_)
        | ExprKind::This => {}
    }
}

fn collect_strings_in_statements(statements: &[Stmt], out: &mut Vec<Vec<u8>>) {
    for statement in statements {
        collect_strings_stmt(statement, out)
    }
}

fn collect_strings_stmt(stmt: &Stmt, out: &mut Vec<Vec<u8>>) {
    match &stmt.kind {
        StmtKind::Variable { initializer, .. } => {
            if let Some(expr) = initializer {
                collect_strings_expr(expr, out)
            }
        }
        StmtKind::Class(class) => {
            if let Some(superclass) = &class.superclass {
                collect_strings_expr(superclass, out)
            }
            for method in &class.methods {
                if let Some(body) = &method.body {
                    match &body.kind {
                        FunctionBodyKind::Expression(expr) => collect_strings_expr(expr, out),
                        FunctionBodyKind::Statements(statements) => collect_strings_in_statements(statements, out),
                    }
                }
            }
        }
        StmtKind::Import(_) | StmtKind::Break | StmtKind::Continue => {}
        StmtKind::Block(statements) => collect_strings_in_statements(statements, out),
        StmtKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_strings_expr(condition, out);
            collect_strings_stmt(then_branch, out);
            if let Some(else_branch) = else_branch {
                collect_strings_stmt(else_branch, out)
            }
        }
        StmtKind::While { condition, body } => {
            collect_strings_expr(condition, out);
            collect_strings_stmt(body, out)
        }
        StmtKind::For { iterable, body, .. } => {
            collect_strings_expr(iterable, out);
            collect_strings_stmt(body, out)
        }
        StmtKind::Return(value) => {
            if let Some(value) = value {
                collect_strings_expr(value, out)
            }
        }
        StmtKind::Expression(expr) => collect_strings_expr(expr, out),
    }
}

fn collect_strings_expr(expr: &Expr, out: &mut Vec<Vec<u8>>) {
    match &expr.kind {
        ExprKind::String(bytes) => out.push(bytes.clone()),
        ExprKind::InterpolatedString(parts) => {
            for part in parts {
                match part {
                    StringPart::Bytes(bytes) => out.push(bytes.clone()),
                    StringPart::Expression(expr) => collect_strings_expr(expr, out),
                }
            }
        }
        ExprKind::List(items) => {
            for item in items {
                collect_strings_expr(item, out)
            }
        }
        ExprKind::Map(entries) => {
            for entry in entries {
                collect_strings_expr(&entry.key, out);
                collect_strings_expr(&entry.value, out)
            }
        }
        ExprKind::Grouped(inner) => collect_strings_expr(inner, out),
        ExprKind::Unary { operand, .. } => collect_strings_expr(operand, out),
        ExprKind::Binary { left, right, .. } | ExprKind::Logical { left, right, .. } => {
            collect_strings_expr(left, out);
            collect_strings_expr(right, out)
        }
        ExprKind::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            collect_strings_expr(condition, out);
            collect_strings_expr(then_expr, out);
            collect_strings_expr(else_expr, out)
        }
        ExprKind::Assign { target, value } => {
            collect_strings_expr(target, out);
            collect_strings_expr(value, out)
        }
        ExprKind::Invoke(invoke) => {
            if let Some(receiver) = &invoke.receiver {
                collect_strings_expr(receiver, out)
            }
            for arg in &invoke.args {
                collect_strings_expr(arg, out)
            }
        }
        ExprKind::Subscript { receiver, args } => {
            collect_strings_expr(receiver, out);
            for arg in args {
                collect_strings_expr(arg, out)
            }
        }
        ExprKind::Await(value) => collect_strings_expr(value, out),
        ExprKind::Closure(body) | ExprKind::AsyncClosure(body) => match &body.kind {
            FunctionBodyKind::Expression(expr) => collect_strings_expr(expr, out),
            FunctionBodyKind::Statements(statements) => collect_strings_in_statements(statements, out),
        },
        ExprKind::Super(super_expr) => {
            for arg in &super_expr.args {
                collect_strings_expr(arg, out)
            }
        }
        ExprKind::Null
        | ExprKind::Bool(_)
        | ExprKind::Number(_)
        | ExprKind::Name(_)
        | ExprKind::Field(_)
        | ExprKind::StaticField(_)
        | ExprKind::This => {}
    }
}

fn collect_import_strings(statements: &[Stmt], out: &mut Vec<String>) {
    for statement in statements {
        match &statement.kind {
            StmtKind::Import(import) => {
                if let Ok(module) = std::str::from_utf8(&import.module) {
                    out.push(module.to_owned());
                }
                for binding in &import.bindings {
                    out.push(binding.imported.value.clone());
                }
            }
            StmtKind::Block(statements) => collect_import_strings(statements, out),
            StmtKind::If {
                then_branch,
                else_branch,
                ..
            } => {
                collect_import_strings(std::slice::from_ref(then_branch.as_ref()), out);
                if let Some(else_branch) = else_branch {
                    collect_import_strings(std::slice::from_ref(else_branch.as_ref()), out)
                }
            }
            StmtKind::While { body, .. } | StmtKind::For { body, .. } => {
                collect_import_strings(std::slice::from_ref(body.as_ref()), out)
            }
            _ => {}
        }
    }
}

fn collect_written_bindings_stmt(
    stmt: &Stmt,
    analysis: &RootAnalysis,
    out: &mut HashSet<BindingId>,
) {
    match &stmt.kind {
        StmtKind::Variable { initializer, .. } => {
            if let Some(initializer) = initializer {
                collect_written_bindings_expr(initializer, analysis, out)
            }
        }
        StmtKind::Class(class) => {
            if let Some(superclass) = &class.superclass {
                collect_written_bindings_expr(superclass, analysis, out)
            }
        }
        StmtKind::Import(_) | StmtKind::Break | StmtKind::Continue => {}
        StmtKind::Block(statements) => {
            for statement in statements {
                collect_written_bindings_stmt(statement, analysis, out)
            }
        }
        StmtKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_written_bindings_expr(condition, analysis, out);
            collect_written_bindings_stmt(then_branch, analysis, out);
            if let Some(else_branch) = else_branch {
                collect_written_bindings_stmt(else_branch, analysis, out)
            }
        }
        StmtKind::While { condition, body } => {
            collect_written_bindings_expr(condition, analysis, out);
            collect_written_bindings_stmt(body, analysis, out)
        }
        StmtKind::For { iterable, body, .. } => {
            collect_written_bindings_expr(iterable, analysis, out);
            collect_written_bindings_stmt(body, analysis, out)
        }
        StmtKind::Return(value) => {
            if let Some(value) = value {
                collect_written_bindings_expr(value, analysis, out)
            }
        }
        StmtKind::Expression(expr) => collect_written_bindings_expr(expr, analysis, out),
    }
}

fn collect_written_bindings_expr(
    expr: &Expr,
    analysis: &RootAnalysis,
    out: &mut HashSet<BindingId>,
) {
    match &expr.kind {
        ExprKind::Assign { target, value } => {
            if let ExprKind::Name(name) = &target.kind {
                if let Some(binding) = analysis.use_map.get(&name.span).copied() {
                    out.insert(binding);
                }
            }
            collect_written_bindings_expr(target, analysis, out);
            collect_written_bindings_expr(value, analysis, out)
        }
        ExprKind::InterpolatedString(parts) => {
            for part in parts {
                if let StringPart::Expression(expr) = part {
                    collect_written_bindings_expr(expr, analysis, out)
                }
            }
        }
        ExprKind::List(items) => {
            for item in items {
                collect_written_bindings_expr(item, analysis, out)
            }
        }
        ExprKind::Map(entries) => {
            for entry in entries {
                collect_written_bindings_expr(&entry.key, analysis, out);
                collect_written_bindings_expr(&entry.value, analysis, out)
            }
        }
        ExprKind::Grouped(inner) => collect_written_bindings_expr(inner, analysis, out),
        ExprKind::Unary { operand, .. } => collect_written_bindings_expr(operand, analysis, out),
        ExprKind::Binary { left, right, .. } | ExprKind::Logical { left, right, .. } => {
            collect_written_bindings_expr(left, analysis, out);
            collect_written_bindings_expr(right, analysis, out)
        }
        ExprKind::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            collect_written_bindings_expr(condition, analysis, out);
            collect_written_bindings_expr(then_expr, analysis, out);
            collect_written_bindings_expr(else_expr, analysis, out)
        }
        ExprKind::Invoke(invoke) => {
            if let Some(receiver) = &invoke.receiver {
                collect_written_bindings_expr(receiver, analysis, out)
            }
            for arg in &invoke.args {
                collect_written_bindings_expr(arg, analysis, out)
            }
        }
        ExprKind::Subscript { receiver, args } => {
            collect_written_bindings_expr(receiver, analysis, out);
            for arg in args {
                collect_written_bindings_expr(arg, analysis, out)
            }
        }
        ExprKind::Super(super_expr) => {
            for arg in &super_expr.args {
                collect_written_bindings_expr(arg, analysis, out)
            }
        }
        // Do not descend into closures: their writes happen when the closure is
        // invoked, and captured mutable bindings are cells already.
        ExprKind::Await(value) => collect_written_bindings_expr(value, analysis, out),
        ExprKind::Closure(_)
        | ExprKind::AsyncClosure(_)
        | ExprKind::Null
        | ExprKind::Bool(_)
        | ExprKind::Number(_)
        | ExprKind::String(_)
        | ExprKind::Name(_)
        | ExprKind::Field(_)
        | ExprKind::StaticField(_)
        | ExprKind::This => {}
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> ModuleAst {
        let parsed = front::parse("test.wren", source);
        assert!(
            !parsed.has_errors(),
            "frontend diagnostics: {:#?}",
            parsed.diagnostics
        );
        parsed.module
    }

    #[test]
    fn mutable_captured_local_becomes_cell() {
        let source = r#"
var f = null
{
  var x = 1
  f = { x = x + 1 }
}
"#;
        let module = parse(source);
        let (analysis, diagnostics) = Analyzer::module(&module, false);
        assert!(diagnostics.is_empty());
        assert!(analysis.bindings.iter().any(|binding| {
            binding.name == "x" && binding.captured && binding.mutated
        }));
    }

    #[test]
    fn immutable_capture_stays_by_value() {
        let source = r#"
{
  var x = 1
  var f = { x }
}
"#;
        let module = parse(source);
        let (analysis, diagnostics) = Analyzer::module(&module, false);
        assert!(diagnostics.is_empty());
        assert!(analysis.bindings.iter().any(|binding| {
            binding.name == "x" && binding.captured && !binding.mutated
        }));
    }

    #[test]
    fn lowers_simple_module_control_flow() {
        let source = r#"
var a = 1
var b = 2
if (a < b) {
  a = b
}
"#;
        let module = parse(source);
        let output = lower_new_program("test.wren", source, "test", &module);
        assert!(
            !output.has_errors(),
            "lower diagnostics: {:#?}",
            output.diagnostics
        );
        assert!(output.lowered.is_some());
    }

    #[test]
    fn class_method_and_closure_are_lowered() {
        let source = r#"
class Counter {
  construct new(v) { _v = v }
  inc() {
    var add = { |n| _v = _v + n }
    add.call(1)
    return _v
  }
}
"#;
        let module = parse(source);
        let output = lower_new_program("test.wren", source, "test", &module);
        assert!(
            !output.has_errors(),
            "lower diagnostics: {:#?}",
            output.diagnostics
        );
        assert!(output.program.functions.len() >= 3);
    }
    #[test]
    fn instance_field_slots_are_local_to_declaring_class() {
        let source = r#"
class Parent {
  parent { _parent }
}

class Child is Parent {
  construct new(v) { _child = v }
  child { _child }
}
"#;
        let module = parse(source);
        let output = lower_new_program("test.wren", source, "test", &module);
        assert!(
            !output.has_errors(),
            "lower diagnostics: {:#?}",
            output.diagnostics
        );

        let child = output
            .program
            .classes
            .iter()
            .find(|class| output.program.string(class.name) == Some("Child"))
            .expect("Child class");
        assert_eq!(child.own_instance_field_count, 1);

        let mut saw_child_field = false;
        for function in &output.program.functions {
            if function.field_owner != Some(child.id) {
                continue;
            }
            for node in &function.nodes {
                match &node.kind {
                    Inst::LoadField { slot, .. } | Inst::StoreField { slot, .. } => {
                        assert_eq!(
                            *slot,
                            FieldSlot(0),
                            "Child's own field must stay class-local before runtime relocation"
                        );
                        saw_child_field = true;
                    }
                    _ => {}
                }
            }
        }
        assert!(saw_child_field);
    }

    #[test]
    fn dynamic_superclass_with_fields_no_longer_requires_static_layout() {
        let source = r#"
class Parent {
  parent { _parent }
}

var Alias = Parent

class Child is Alias {
  construct new(v) { _child = v }
  child { _child }
}
"#;
        let module = parse(source);
        let output = lower_new_program("test.wren", source, "test", &module);
        assert!(
            !output.has_errors(),
            "dynamic superclass field layout should be relocated at runtime: {:#?}",
            output.diagnostics
        );

        let child = output
            .program
            .classes
            .iter()
            .find(|class| output.program.string(class.name) == Some("Child"))
            .expect("Child class");
        assert_eq!(child.own_instance_field_count, 1);
        assert_eq!(
            child.superclass, None,
            "Alias is intentionally dynamic from the lowerer's point of view"
        );
    }

}
