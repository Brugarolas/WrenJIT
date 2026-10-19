//! Wren VM executable entry point.
//!
//! Expected layout:
//!
//! ```text
//! src/
//! ├── main.rs
//! ├── wren_front.rs
//! ├── wren_builtin.rs
//! ├── wren_lowerer.rs
//! ├── wren_ir.rs
//! ├── wren_bytecode.rs
//! ├── wren_bytecodecompiler.rs
//! └── wren_interpreter.rs
//! ```
//!
//! Usage:
//!
//! ```text
//! cargo run -- path/to/program.wren
//! ```
//!
//! The pipeline is:
//!
//! ```text
//! source
//!   -> tokenizer/parser
//!   -> AST
//!   -> Wren HIR
//!   -> register bytecode
//!   -> interpreter
//! ```

pub mod wren_builtin;
pub mod wren_bytecode;
pub mod wren_bytecodecompiler;
pub mod wren_front;
pub mod wren_interpreter;
pub mod wren_ir;
pub mod wren_lowerer;
pub mod wren_traceir;
pub mod wren_tracingjit;

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use crate::wren_interpreter::{ExecutionResult, Interpreter};
use crate::wren_ir::Program;

const EXIT_SUCCESS: u8 = 0;
const EXIT_COMPILE_ERROR: u8 = 1;
const EXIT_USAGE_ERROR: u8 = 2;
const EXIT_IO_ERROR: u8 = 3;
const EXIT_RUNTIME_ERROR: u8 = 4;

fn main() -> ExitCode {
    ExitCode::from(run())
}

fn run() -> u8 {
    let source_path = match parse_command_line() {
        Ok(path) => path,
        Err(()) => return EXIT_USAGE_ERROR,
    };

    let source = match fs::read_to_string(&source_path) {
        Ok(source) => source,
        Err(error) => {
            eprintln!(
                "error: failed to read '{}': {error}",
                source_path.display()
            );
            return EXIT_IO_ERROR;
        }
    };

    let source_name = source_path.to_string_lossy();
    let module_name = module_name_from_path(&source_path);

    // -------------------------------------------------------------------------
    // Frontend: source -> AST
    // -------------------------------------------------------------------------

    let parsed = wren_front::parse(&source_name, &source);

    if let Err(error) = parsed.emit(&source_name, &source) {
        eprintln!("error: failed to render parser diagnostics: {error}");
        return EXIT_IO_ERROR;
    }

    if parsed.has_errors() {
        return EXIT_COMPILE_ERROR;
    }

    // -------------------------------------------------------------------------
    // Lowering: AST -> SSA HIR
    // -------------------------------------------------------------------------

    let mut hir = Program::new();
    let builtin_manifest = wren_builtin::install_hir_builtins(&mut hir);
    let lowered = wren_lowerer::lower_module(
        &mut hir,
        &source_name,
        &source,
        &module_name,
        &parsed.module,
    );

    if let Err(error) = lowered.emit(&source_name, &source) {
        eprintln!("error: failed to render lowering diagnostics: {error}");
        return EXIT_IO_ERROR;
    }

    if lowered.has_errors() {
        return EXIT_COMPILE_ERROR;
    }

    let lowered_module = match lowered.lowered {
        Some(module) => module,
        None => {
            eprintln!("error: lowering completed without producing a module");
            return EXIT_COMPILE_ERROR;
        }
    };

    // -------------------------------------------------------------------------
    // Backend: HIR -> register bytecode
    // -------------------------------------------------------------------------

    let bytecode = match wren_bytecodecompiler::compile_program(&hir) {
        Ok(program) => program,
        Err(error) => {
            eprintln!("bytecode compilation error: {error}");
            return EXIT_COMPILE_ERROR;
        }
    };

    // -------------------------------------------------------------------------
    // Runtime: bytecode -> execution
    // -------------------------------------------------------------------------

    let host = Box::new(wren_builtin::BuiltinHost::new(builtin_manifest.clone()));
    let mut interpreter = match Interpreter::with_host(&bytecode, host) {
        Ok(interpreter) => interpreter,
        Err(error) => {
            eprintln!("VM initialization error: {error}");
            return EXIT_RUNTIME_ERROR;
        }
    };
    wren_builtin::install_vm_builtins(&mut interpreter, &builtin_manifest);

    match interpreter.run_module(lowered_module.entry) {
        Ok(ExecutionResult::Returned(_)) => EXIT_SUCCESS,

        Ok(ExecutionResult::Suspended(_)) => {
            eprintln!(
                "runtime error: top-level module '{}' suspended instead of returning",
                module_name
            );
            EXIT_RUNTIME_ERROR
        }

        Err(error) => {
            eprintln!("runtime error: {error}");
            EXIT_RUNTIME_ERROR
        }
    }
}

fn parse_command_line() -> Result<PathBuf, ()> {
    let mut args = env::args_os();
    let executable = args
        .next()
        .unwrap_or_else(|| OsStr::new("wren").to_owned());

    let executable = Path::new(&executable)
        .file_name()
        .unwrap_or_else(|| OsStr::new("wren"))
        .to_string_lossy();

    let Some(first) = args.next() else {
        print_usage(&executable);
        return Err(());
    };

    if first == OsStr::new("-h") || first == OsStr::new("--help") {
        print_usage(&executable);
        return Err(());
    }

    if let Some(extra) = args.next() {
        eprintln!(
            "error: unexpected extra argument '{}'",
            extra.to_string_lossy()
        );
        print_usage(&executable);
        return Err(());
    }

    Ok(PathBuf::from(first))
}

fn print_usage(executable: &str) {
    eprintln!("Usage: {executable} <file.wren>");
}

fn module_name_from_path(path: &Path) -> String {
    path.file_stem()
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| OsStr::new("main"))
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_module_name_from_file_name() {
        assert_eq!(
            module_name_from_path(Path::new("examples/hello.wren")),
            "hello"
        );
    }

    #[test]
    fn derives_fallback_module_name() {
        assert_eq!(module_name_from_path(Path::new("")), "main");
    }
}
