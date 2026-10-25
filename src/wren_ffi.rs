//! Dynamic C FFI support for WrenJIT.
//!
//! The managed Wren heap stores only Send/Sync metadata (`FfiLibrary` and
//! `FfiFunction`). Prepared libffi CIFs deliberately live in a thread-local
//! signature cache: libffi's safe middle-layer `Cif` is !Send/!Sync. This keeps
//! that contract intact while allowing an Interpreter/isolate to move between
//! scheduler workers; each worker prepares a signature at most once.
//!
//! This module only implements the platform C ABI. Callbacks, variadics and
//! structs-by-value are intentionally outside the first public FFI surface.

use libffi::middle::{Arg, Cif, CodePtr, Type};
use libloading::Library;

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{
    CString, c_char, c_int, c_long, c_longlong, c_short, c_uchar, c_uint, c_ulong,
    c_ulonglong, c_ushort, c_void,
};
use std::fmt;
use std::sync::Arc;

pub const MAX_FFI_ARITY: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FfiType {
    Void,
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    Isize,
    Usize,
    CChar,
    CUChar,
    CShort,
    CUShort,
    CInt,
    CUInt,
    CLong,
    CULong,
    CLongLong,
    CULongLong,
    F32,
    F64,
    Pointer,
    CString,
}

impl FfiType {
    pub fn parse(name: &str) -> Result<Self, FfiError> {
        let normalized = name.trim().to_ascii_lowercase().replace([' ', '\t'], "");
        let ty = match normalized.as_str() {
            "void" => Self::Void,
            "bool" | "_bool" => Self::Bool,
            "i8" | "int8" | "int8_t" => Self::I8,
            "u8" | "uint8" | "uint8_t" | "byte" => Self::U8,
            "i16" | "int16" | "int16_t" => Self::I16,
            "u16" | "uint16" | "uint16_t" => Self::U16,
            "i32" | "int32" | "int32_t" => Self::I32,
            "u32" | "uint32" | "uint32_t" => Self::U32,
            "i64" | "int64" | "int64_t" => Self::I64,
            "u64" | "uint64" | "uint64_t" => Self::U64,
            "isize" | "ssize_t" | "intptr_t" | "ptrdiff_t" => Self::Isize,
            "usize" | "size_t" | "uintptr_t" => Self::Usize,
            "char" | "c_char" => Self::CChar,
            "schar" | "signedchar" | "c_schar" => Self::I8,
            "uchar" | "unsignedchar" | "c_uchar" => Self::CUChar,
            "short" | "shortint" | "signedshort" | "c_short" => Self::CShort,
            "ushort" | "unsignedshort" | "c_ushort" => Self::CUShort,
            "int" | "signedint" | "c_int" => Self::CInt,
            "uint" | "unsignedint" | "c_uint" => Self::CUInt,
            "long" | "longint" | "signedlong" | "c_long" => Self::CLong,
            "ulong" | "unsignedlong" | "c_ulong" => Self::CULong,
            "longlong" | "longlongint" | "signedlonglong" | "c_longlong" => Self::CLongLong,
            "ulonglong" | "unsignedlonglong" | "c_ulonglong" => Self::CULongLong,
            "f32" | "float" => Self::F32,
            "f64" | "double" => Self::F64,
            "ptr" | "pointer" | "void*" | "constvoid*" => Self::Pointer,
            "cstring" | "char*" | "constchar*" | "cstr" => Self::CString,
            _ => return Err(FfiError::new(format!("unknown FFI type `{name}`"))),
        };
        Ok(ty)
    }

    #[inline]
    pub const fn is_void(self) -> bool { matches!(self, Self::Void) }

    #[inline]
    pub const fn is_pointer(self) -> bool { matches!(self, Self::Pointer | Self::CString) }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Void => "void",
            Self::Bool => "bool",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::U64 => "u64",
            Self::Isize => "isize",
            Self::Usize => "usize",
            Self::CChar => "char",
            Self::CUChar => "unsigned char",
            Self::CShort => "short",
            Self::CUShort => "unsigned short",
            Self::CInt => "int",
            Self::CUInt => "unsigned int",
            Self::CLong => "long",
            Self::CULong => "unsigned long",
            Self::CLongLong => "long long",
            Self::CULongLong => "unsigned long long",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Pointer => "pointer",
            Self::CString => "cstring",
        }
    }

    fn libffi_type(self) -> Type {
        match self {
            Self::Void => Type::void(),
            Self::Bool => Type::u8(),
            Self::I8 => Type::i8(),
            Self::U8 => Type::u8(),
            Self::I16 => Type::i16(),
            Self::U16 => Type::u16(),
            Self::I32 => Type::i32(),
            Self::U32 => Type::u32(),
            Self::I64 => Type::i64(),
            Self::U64 => Type::u64(),
            Self::Isize => Type::isize(),
            Self::Usize => Type::usize(),
            Self::CChar => {
                if (u8::MAX as c_char) < 0 { Type::c_schar() } else { Type::c_uchar() }
            },
            Self::CUChar => Type::c_uchar(),
            Self::CShort => Type::c_short(),
            Self::CUShort => Type::c_ushort(),
            Self::CInt => Type::c_int(),
            Self::CUInt => Type::c_uint(),
            Self::CLong => Type::c_long(),
            Self::CULong => Type::c_ulong(),
            Self::CLongLong => Type::c_longlong(),
            Self::CULongLong => Type::c_ulonglong(),
            Self::F32 => Type::f32(),
            Self::F64 => Type::f64(),
            Self::Pointer | Self::CString => Type::pointer(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FfiSignature {
    pub result: FfiType,
    pub args: Box<[FfiType]>,
}

impl FfiSignature {
    pub fn new(result: FfiType, args: Vec<FfiType>) -> Result<Self, FfiError> {
        if args.len() > MAX_FFI_ARITY {
            return Err(FfiError::new(format!(
                "FFI functions support at most {MAX_FFI_ARITY} arguments"
            )));
        }
        if let Some(index) = args.iter().position(|ty| ty.is_void()) {
            return Err(FfiError::new(format!(
                "FFI argument {index} cannot have type void"
            )));
        }
        Ok(Self { result, args: args.into_boxed_slice() })
    }

    fn prepare(&self) -> Result<Cif, FfiError> {
        Cif::try_new(
            self.args.iter().copied().map(FfiType::libffi_type),
            self.result.libffi_type(),
        )
        .map_err(|error| FfiError::new(format!("libffi rejected signature: {error:?}")))
    }
}

pub struct FfiLibrary {
    name: String,
    library: Library,
}

impl fmt::Debug for FfiLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FfiLibrary").field("name", &self.name).finish_non_exhaustive()
    }
}

impl FfiLibrary {
    pub fn open(path: &str) -> Result<Arc<Self>, FfiError> {
        if path.is_empty() {
            return Err(FfiError::new("FFI.open path must not be empty"));
        }
        let library = unsafe { Library::new(path) }
            .map_err(|error| FfiError::new(format!("failed to open `{path}`: {error}")))?;
        Ok(Arc::new(Self { name: path.to_owned(), library }))
    }

    #[inline]
    pub fn name(&self) -> &str { &self.name }

    pub fn bind(
        self: &Arc<Self>,
        symbol: &str,
        signature: FfiSignature,
    ) -> Result<Arc<FfiFunction>, FfiError> {
        if symbol.is_empty() {
            return Err(FfiError::new("FFI symbol name must not be empty"));
        }
        if symbol.as_bytes().contains(&0) {
            return Err(FfiError::new("FFI symbol name contains an interior NUL byte"));
        }

        // Copy the raw code address while the temporary Symbol lifetime is tied
        // to this Library. FfiFunction retains an Arc to the library, so the
        // address cannot outlive the mapping from which it came.
        let address = unsafe {
            let function = self
                .library
                .get::<unsafe extern "C" fn()>(symbol.as_bytes())
                .map_err(|error| {
                    FfiError::new(format!(
                        "failed to resolve `{symbol}` from `{}`: {error}", self.name
                    ))
                })?;
            *function as usize
        };

        Ok(Arc::new(FfiFunction {
            library: self.clone(),
            symbol: symbol.to_owned(),
            address,
            signature,
        }))
    }
}

pub struct FfiFunction {
    // Keeps the dynamic library loaded for at least as long as the code address.
    library: Arc<FfiLibrary>,
    symbol: String,
    address: usize,
    signature: FfiSignature,
}

impl fmt::Debug for FfiFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FfiFunction")
            .field("library", &self.library.name())
            .field("symbol", &self.symbol)
            .field("address", &format_args!("0x{:x}", self.address))
            .field("signature", &self.signature)
            .finish()
    }
}

impl FfiFunction {
    #[inline]
    pub fn library(&self) -> &Arc<FfiLibrary> { &self.library }
    #[inline]
    pub fn symbol(&self) -> &str { &self.symbol }
    #[inline]
    pub fn signature(&self) -> &FfiSignature { &self.signature }
    #[inline]
    pub fn address(&self) -> usize { self.address }
}

/// Dynamic-library lifetime and last-error state owned by one interpreter.
/// Prepared CIFs are thread-local because libffi intentionally marks Cif as
/// !Send/!Sync. A migrated isolate simply warms the signature on its new worker.
#[derive(Default)]
pub struct FfiRuntime {
    // Keep dynamic libraries mapped until Interpreter teardown. This ensures that
    // GC worker threads can drop managed FFI wrappers without being the thread
    // that executes dlclose/FreeLibrary and any library finalizers it may run.
    libraries: Vec<Arc<FfiLibrary>>,
    last_error: i32,
}

thread_local! {
    static FFI_CIF_CACHE: RefCell<HashMap<FfiSignature, Cif>> = RefCell::new(HashMap::new());
}

impl fmt::Debug for FfiRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FfiRuntime")
            .field("retained_libraries", &self.libraries.len())
            .field("last_error", &self.last_error)
            .finish()
    }
}

impl FfiRuntime {
    pub fn retain_library(&mut self, library: Arc<FfiLibrary>) {
        if !self.libraries.iter().any(|existing| Arc::ptr_eq(existing, &library)) {
            self.libraries.push(library);
        }
    }

    pub fn prepare(&mut self, signature: &FfiSignature) -> Result<(), FfiError> {
        FFI_CIF_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if !cache.contains_key(signature) {
                cache.insert(signature.clone(), signature.prepare()?);
            }
            Ok(())
        })
    }

    #[inline]
    pub fn last_error(&self) -> i32 { self.last_error }

    pub fn call(
        &mut self,
        function: &FfiFunction,
        args: &[NativeArg],
    ) -> Result<NativeResult, FfiError> {
        let signature = function.signature();
        if args.len() != signature.args.len() {
            return Err(FfiError::new(format!(
                "{} expects {} FFI arguments, got {}",
                function.symbol(), signature.args.len(), args.len()
            )));
        }

        let ffi_args: Vec<Arg<'_>> = args.iter().map(NativeArg::as_ffi_arg).collect();
        let code = CodePtr(function.address() as *mut c_void);

        let (result, last_error) = FFI_CIF_CACHE.with(|cache| -> Result<_, FfiError> {
            let mut cache = cache.borrow_mut();
            if !cache.contains_key(signature) {
                cache.insert(signature.clone(), signature.prepare()?);
            }
            let cif = cache.get(signature).expect("CIF was prepared above");

            // SAFETY: the public FFI binder constructs `signature` and every
            // NativeArg from the same declared FfiType list. The remaining unsafe
            // contract is inherent to any C FFI: the user must declare a signature
            // matching the actual native symbol.
            let result = unsafe {
                match signature.result {
                    FfiType::Void => { cif.call::<()>(code, &ffi_args); NativeResult::Void }
                    FfiType::Bool => NativeResult::Bool(cif.call::<u8>(code, &ffi_args) != 0),
                    FfiType::I8 => NativeResult::I8(cif.call::<i8>(code, &ffi_args)),
                    FfiType::U8 => NativeResult::U8(cif.call::<u8>(code, &ffi_args)),
                    FfiType::I16 => NativeResult::I16(cif.call::<i16>(code, &ffi_args)),
                    FfiType::U16 => NativeResult::U16(cif.call::<u16>(code, &ffi_args)),
                    FfiType::I32 => NativeResult::I32(cif.call::<i32>(code, &ffi_args)),
                    FfiType::U32 => NativeResult::U32(cif.call::<u32>(code, &ffi_args)),
                    FfiType::I64 => NativeResult::I64(cif.call::<i64>(code, &ffi_args)),
                    FfiType::U64 => NativeResult::U64(cif.call::<u64>(code, &ffi_args)),
                    FfiType::Isize => NativeResult::Isize(cif.call::<isize>(code, &ffi_args)),
                    FfiType::Usize => NativeResult::Usize(cif.call::<usize>(code, &ffi_args)),
                    FfiType::CChar => NativeResult::CChar(cif.call::<c_char>(code, &ffi_args)),
                    FfiType::CUChar => NativeResult::CUChar(cif.call::<c_uchar>(code, &ffi_args)),
                    FfiType::CShort => NativeResult::CShort(cif.call::<c_short>(code, &ffi_args)),
                    FfiType::CUShort => NativeResult::CUShort(cif.call::<c_ushort>(code, &ffi_args)),
                    FfiType::CInt => NativeResult::CInt(cif.call::<c_int>(code, &ffi_args)),
                    FfiType::CUInt => NativeResult::CUInt(cif.call::<c_uint>(code, &ffi_args)),
                    FfiType::CLong => NativeResult::CLong(cif.call::<c_long>(code, &ffi_args)),
                    FfiType::CULong => NativeResult::CULong(cif.call::<c_ulong>(code, &ffi_args)),
                    FfiType::CLongLong => NativeResult::CLongLong(cif.call::<c_longlong>(code, &ffi_args)),
                    FfiType::CULongLong => NativeResult::CULongLong(cif.call::<c_ulonglong>(code, &ffi_args)),
                    FfiType::F32 => NativeResult::F32(cif.call::<f32>(code, &ffi_args)),
                    FfiType::F64 => NativeResult::F64(cif.call::<f64>(code, &ffi_args)),
                    FfiType::Pointer | FfiType::CString => {
                        NativeResult::Pointer(cif.call::<*mut c_void>(code, &ffi_args))
                    }
                }
            };

            // Capture errno/GetLastError before even releasing the TLS cache
            // borrow; no Wren allocation or host operation can occur first.
            let last_error = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            Ok((result, last_error))
        })?;

        self.last_error = last_error;
        Ok(result)
    }
}

#[derive(Debug)]
pub enum NativeArg {
    Bool(u8),
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    Isize(isize),
    Usize(usize),
    CChar(c_char),
    CUChar(c_uchar),
    CShort(c_short),
    CUShort(c_ushort),
    CInt(c_int),
    CUInt(c_uint),
    CLong(c_long),
    CULong(c_ulong),
    CLongLong(c_longlong),
    CULongLong(c_ulonglong),
    F32(f32),
    F64(f64),
    Pointer(*mut c_void),
    CString { storage: CString, pointer: *const c_char },
}

impl NativeArg {
    pub fn cstring(bytes: &[u8]) -> Result<Self, FfiError> {
        let storage = CString::new(bytes)
            .map_err(|_| FfiError::new("cstring argument contains an interior NUL byte"))?;
        let pointer = storage.as_ptr();
        Ok(Self::CString { storage, pointer })
    }

    fn as_ffi_arg(&self) -> Arg<'_> {
        match self {
            Self::Bool(value) => Arg::new(value),
            Self::I8(value) => Arg::new(value),
            Self::U8(value) => Arg::new(value),
            Self::I16(value) => Arg::new(value),
            Self::U16(value) => Arg::new(value),
            Self::I32(value) => Arg::new(value),
            Self::U32(value) => Arg::new(value),
            Self::I64(value) => Arg::new(value),
            Self::U64(value) => Arg::new(value),
            Self::Isize(value) => Arg::new(value),
            Self::Usize(value) => Arg::new(value),
            Self::CChar(value) => Arg::new(value),
            Self::CUChar(value) => Arg::new(value),
            Self::CShort(value) => Arg::new(value),
            Self::CUShort(value) => Arg::new(value),
            Self::CInt(value) => Arg::new(value),
            Self::CUInt(value) => Arg::new(value),
            Self::CLong(value) => Arg::new(value),
            Self::CULong(value) => Arg::new(value),
            Self::CLongLong(value) => Arg::new(value),
            Self::CULongLong(value) => Arg::new(value),
            Self::F32(value) => Arg::new(value),
            Self::F64(value) => Arg::new(value),
            Self::Pointer(value) => Arg::new(value),
            Self::CString { pointer, storage } => {
                // Keep an explicit read so the ownership field is visibly live
                // for the entire Arg borrow; the C call receives `pointer`.
                let _keep_alive = storage;
                Arg::new(pointer)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum NativeResult {
    Void,
    Bool(bool),
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    Isize(isize),
    Usize(usize),
    CChar(c_char),
    CUChar(c_uchar),
    CShort(c_short),
    CUShort(c_ushort),
    CInt(c_int),
    CUInt(c_uint),
    CLong(c_long),
    CULong(c_ulong),
    CLongLong(c_longlong),
    CULongLong(c_ulonglong),
    F32(f32),
    F64(f64),
    Pointer(*mut c_void),
}

#[derive(Clone, Debug)]
pub struct FfiError {
    pub message: String,
}

impl FfiError {
    #[inline]
    pub fn new(message: impl Into<String>) -> Self { Self { message: message.into() } }
}

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.message) }
}

impl std::error::Error for FfiError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_c_aliases() {
        assert_eq!(FfiType::parse("size_t").unwrap(), FfiType::Usize);
        assert_eq!(FfiType::parse("const char *").unwrap(), FfiType::CString);
        assert_eq!(FfiType::parse("unsigned long long").unwrap(), FfiType::CULongLong);
        assert!(FfiType::parse("not_a_type").is_err());
        assert!(FfiSignature::new(FfiType::Void, vec![FfiType::Void]).is_err());
    }

    #[test]
    fn cstring_rejects_embedded_nul() {
        assert!(NativeArg::cstring(b"hello").is_ok());
        assert!(NativeArg::cstring(b"hel\0lo").is_err());
    }

    #[test]
    fn libffi_scalar_call_matches_native_function() {
        extern "C" fn add(left: i32, right: i32) -> i32 { left + right }
        let signature = FfiSignature::new(FfiType::I32, vec![FfiType::I32, FfiType::I32]).unwrap();
        let cif = signature.prepare().unwrap();
        let left = 20i32;
        let right = 22i32;
        let args = [Arg::new(&left), Arg::new(&right)];
        let value: i32 = unsafe { cif.call(CodePtr(add as *mut _), &args) };
        assert_eq!(value, 42);
    }

    #[test]
    fn libffi_f64_call_matches_native_function() {
        extern "C" fn affine(value: f64, scale: f64, bias: f64) -> f64 {
            value * scale + bias
        }
        let signature = FfiSignature::new(
            FfiType::F64,
            vec![FfiType::F64, FfiType::F64, FfiType::F64],
        ).unwrap();
        let cif = signature.prepare().unwrap();
        let value = 10.0f64;
        let scale = 4.0f64;
        let bias = 2.0f64;
        let args = [Arg::new(&value), Arg::new(&scale), Arg::new(&bias)];
        let result: f64 = unsafe { cif.call(CodePtr(affine as *mut _), &args) };
        assert_eq!(result, 42.0);
    }
}
