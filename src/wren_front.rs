//! Wren source frontend: tokenizer, AST, parser, and diagnostics.
//!
//! Dependencies:
//! ```toml
//! logos = "0.16"
//! ariadne = "0.6"
//! ```
//!
//! Design goals:
//! - Faithful Wren syntax, including significant newlines.
//! - Fast ordinary lexing through `logos`.
//! - Correct nested block comments and nested string interpolation.
//! - Byte-accurate spans throughout the compiler frontend.
//! - A lossless-enough AST for a later Wren IR lowerer.
//! - Structured diagnostics rendered with `ariadne`.
//! - Parser error recovery so a single typo does not hide later errors.

#![allow(clippy::large_enum_variant)]
#![allow(clippy::too_many_arguments)]

use std::io;
use std::ops::Range;

use ariadne::{Color, Label, Report, ReportKind, Source};
use logos::Logos;

pub const MAX_PARAMETERS: usize = 16;
pub const MAX_INTERPOLATION_DEPTH: usize = 8;
pub const MAX_PARSE_DEPTH: usize = 256;

// =============================================================================
// Source spans and diagnostics
// =============================================================================

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    #[inline]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    #[inline]
    pub const fn empty(at: usize) -> Self {
        Self { start: at, end: at }
    }

    #[inline]
    pub const fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }

    #[inline]
    pub const fn join(self, other: Span) -> Span {
        Span {
            start: if self.start < other.start { self.start } else { other.start },
            end: if self.end > other.end { self.end } else { other.end },
        }
    }

    #[inline]
    pub fn as_range(self) -> Range<usize> {
        self.start..self.end
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spanned<T> {
    pub value: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    #[inline]
    pub const fn new(value: T, span: Span) -> Self {
        Self { value, span }
    }

    #[inline]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned {
            value: f(self.value),
            span: self.span,
        }
    }
}

pub type Ident = Spanned<String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticLabel {
    pub span: Span,
    pub message: String,
    pub primary: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: DiagnosticSeverity,
    pub code: &'static str,
    pub message: String,
    pub labels: Vec<DiagnosticLabel>,
    pub notes: Vec<String>,
}

impl Diagnostic {
    pub fn error(code: &'static str, message: impl Into<String>, span: Span) -> Self {
        Self {
            severity: DiagnosticSeverity::Error,
            code,
            message: message.into(),
            labels: vec![DiagnosticLabel {
                span,
                message: String::new(),
                primary: true,
            }],
            notes: Vec::new(),
        }
    }

    pub fn warning(code: &'static str, message: impl Into<String>, span: Span) -> Self {
        Self {
            severity: DiagnosticSeverity::Warning,
            code,
            message: message.into(),
            labels: vec![DiagnosticLabel {
                span,
                message: String::new(),
                primary: true,
            }],
            notes: Vec::new(),
        }
    }

    pub fn with_primary_label(mut self, span: Span, message: impl Into<String>) -> Self {
        let message = message.into();
        if let Some(label) = self
            .labels
            .iter_mut()
            .find(|label| label.primary && label.span == span && label.message.is_empty())
        {
            label.message = message;
            return self;
        }

        self.labels.push(DiagnosticLabel {
            span,
            message,
            primary: true,
        });
        self
    }

    pub fn with_secondary_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(DiagnosticLabel {
            span,
            message: message.into(),
            primary: false,
        });
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Renders this diagnostic to stderr using Ariadne.
    ///
    /// Frontend spans use UTF-8 byte offsets because that is what Logos returns
    /// and what the rest of the compiler wants. Ariadne 0.6 uses character
    /// offsets, so conversion is performed only on the cold diagnostic path.
    pub fn eprint(&self, source_name: &str, source: &str) -> io::Result<()> {
        let anchor = self
            .labels
            .first()
            .map(|label| byte_span_to_char_range(source, label.span))
            .unwrap_or(0..0);

        let kind = match self.severity {
            DiagnosticSeverity::Error => ReportKind::Error,
            DiagnosticSeverity::Warning => ReportKind::Warning,
        };

        let mut report = Report::build(kind, (source_name, anchor))
            .with_code(self.code)
            .with_message(self.message.clone());

        for label in &self.labels {
            let range = byte_span_to_char_range(source, label.span);
            let color = if label.primary { Color::Red } else { Color::Yellow };
            let mut ariadne_label = Label::new((source_name, range)).with_color(color);
            if !label.message.is_empty() {
                ariadne_label = ariadne_label.with_message(label.message.clone());
            }
            report = report.with_label(ariadne_label);
        }

        for note in &self.notes {
            report = report.with_note(note.clone());
        }

        report.finish().eprint((source_name, Source::from(source)))
    }
}

pub fn emit_diagnostics(
    source_name: &str,
    source: &str,
    diagnostics: &[Diagnostic],
) -> io::Result<()> {
    for diagnostic in diagnostics {
        diagnostic.eprint(source_name, source)?;
    }
    Ok(())
}

#[inline]
fn byte_span_to_char_range(source: &str, span: Span) -> Range<usize> {
    byte_offset_to_char_offset(source, span.start)..byte_offset_to_char_offset(source, span.end)
}

#[inline]
fn byte_offset_to_char_offset(source: &str, byte_offset: usize) -> usize {
    let mut clamped = byte_offset.min(source.len());
    while clamped > 0 && !source.is_char_boundary(clamped) {
        clamped -= 1;
    }
    source[..clamped].chars().count()
}

// =============================================================================
// Tokenizer
// =============================================================================

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LexErrorKind {
    UnterminatedBlockComment,
    UnterminatedString,
    UnterminatedRawString,
    InterpolationTooDeep,
    InvalidInterpolation,
    UnterminatedScientificNotation,
    #[default]
    UnexpectedCharacter,
}

#[derive(Logos, Clone, Copy, Debug, PartialEq, Eq)]
#[logos(error = LexErrorKind)]
enum RawTokenKind {
    #[regex(r"[ \t\r]+", logos::skip)]
    _Whitespace,

    #[regex(r"//[^\r\n]*", logos::skip, allow_greedy = true)]
    _LineComment,

    #[token("/*", lex_block_comment)]
    BlockComment,

    #[token("\n")]
    Newline,

    // Keywords.
    #[token("as")]
    As,
    #[token("break")]
    Break,
    #[token("class")]
    Class,
    #[token("construct")]
    Construct,
    #[token("continue")]
    Continue,
    #[token("else")]
    Else,
    #[token("false")]
    False,
    #[token("for")]
    For,
    #[token("foreign")]
    Foreign,
    #[token("if")]
    If,
    #[token("import")]
    Import,
    #[token("in")]
    In,
    #[token("is")]
    Is,
    #[token("null")]
    Null,
    #[token("return")]
    Return,
    #[token("static")]
    Static,
    #[token("super")]
    Super,
    #[token("this")]
    This,
    #[token("true")]
    True,
    #[token("var")]
    Var,
    #[token("while")]
    While,

    // Literals. Triple quotes must win over a normal quote.
    #[token("\"\"\"", lex_raw_string, priority = 4)]
    RawString,
    #[token("\"", lex_normal_string, priority = 3)]
    String,
    #[regex(r"0x[0-9a-fA-F]+", priority = 3)]
    Number,
    #[regex(
        r"[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]*)?",
        lex_decimal_number,
        priority = 2
    )]
    NumberDec,

    // Wren treats leading underscores as field syntax, not ordinary names.
    #[regex(r"__[A-Za-z0-9_]*", priority = 3)]
    StaticField,
    #[regex(r"_[A-Za-z0-9_]*", priority = 2)]
    Field,
    #[regex(r"[A-Za-z][A-Za-z0-9_]*")]
    Identifier,

    // Longest operators first.
    #[token("...")]
    DotDotDot,
    #[token("..")]
    DotDot,
    #[token("==")]
    EqualEqual,
    #[token("!=")]
    BangEqual,
    #[token("<=")]
    LessEqual,
    #[token(">=")]
    GreaterEqual,
    #[token("<<")]
    ShiftLeft,
    #[token(">>")]
    ShiftRight,
    #[token("&&")]
    AndAnd,
    #[token("||")]
    OrOr,

    #[token("(")]
    LeftParen,
    #[token(")")]
    RightParen,
    #[token("[")]
    LeftBracket,
    #[token("]")]
    RightBracket,
    #[token("{")]
    LeftBrace,
    #[token("}")]
    RightBrace,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token(":")]
    Colon,
    #[token("?")]
    Question,
    #[token("=")]
    Equal,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("!")]
    Bang,
    #[token("~")]
    Tilde,
    #[token("&")]
    Ampersand,
    #[token("|")]
    Pipe,
    #[token("^")]
    Caret,
    #[token("<")]
    Less,
    #[token(">")]
    Greater,
    #[token("#")]
    Hash,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenKind {
    Newline,
    As,
    Break,
    Class,
    Construct,
    Continue,
    Else,
    False,
    For,
    Foreign,
    If,
    Import,
    In,
    Is,
    Null,
    Return,
    Static,
    Super,
    This,
    True,
    Var,
    While,
    RawString,
    String,
    Number,
    Field,
    StaticField,
    Identifier,
    DotDotDot,
    DotDot,
    EqualEqual,
    BangEqual,
    LessEqual,
    GreaterEqual,
    ShiftLeft,
    ShiftRight,
    AndAnd,
    OrOr,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Colon,
    Question,
    Equal,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Bang,
    Tilde,
    Ampersand,
    Pipe,
    Caret,
    Less,
    Greater,
    Hash,
    Eof,
}

impl TokenKind {
    pub const fn description(self) -> &'static str {
        use TokenKind::*;
        match self {
            Newline => "newline",
            As => "`as`",
            Break => "`break`",
            Class => "`class`",
            Construct => "`construct`",
            Continue => "`continue`",
            Else => "`else`",
            False => "`false`",
            For => "`for`",
            Foreign => "`foreign`",
            If => "`if`",
            Import => "`import`",
            In => "`in`",
            Is => "`is`",
            Null => "`null`",
            Return => "`return`",
            Static => "`static`",
            Super => "`super`",
            This => "`this`",
            True => "`true`",
            Var => "`var`",
            While => "`while`",
            RawString => "raw string",
            String => "string",
            Number => "number",
            Field => "instance field",
            StaticField => "static field",
            Identifier => "identifier",
            DotDotDot => "`...`",
            DotDot => "`..`",
            EqualEqual => "`==`",
            BangEqual => "`!=`",
            LessEqual => "`<=`",
            GreaterEqual => "`>=`",
            ShiftLeft => "`<<`",
            ShiftRight => "`>>`",
            AndAnd => "`&&`",
            OrOr => "`||`",
            LeftParen => "`(`",
            RightParen => "`)`",
            LeftBracket => "`[`",
            RightBracket => "`]`",
            LeftBrace => "`{`",
            RightBrace => "`}`",
            Comma => "`,`",
            Dot => "`.`",
            Colon => "`:`",
            Question => "`?`",
            Equal => "`=`",
            Plus => "`+`",
            Minus => "`-`",
            Star => "`*`",
            Slash => "`/`",
            Percent => "`%`",
            Bang => "`!`",
            Tilde => "`~`",
            Ampersand => "`&`",
            Pipe => "`|`",
            Caret => "`^`",
            Less => "`<`",
            Greater => "`>`",
            Hash => "`#`",
            Eof => "end of file",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    #[inline]
    pub const fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }
}

#[derive(Debug)]
pub struct LexOutput {
    pub tokens: Vec<Token>,
    pub diagnostics: Vec<Diagnostic>,
}

pub fn lex(source: &str) -> LexOutput {
    let (source, base_offset) = if source.as_bytes().starts_with(&[0xEF, 0xBB, 0xBF]) {
        (&source[3..], 3)
    } else {
        (source, 0)
    };

    // Like the reference implementation, accept a Unix shebang only at the
    // beginning of the source. Start lexing at its terminating newline so the
    // newline remains visible to the parser.
    if source.starts_with("#!/") {
        let skipped = source.find('\n').unwrap_or(source.len());
        lex_with_offset(&source[skipped..], base_offset + skipped)
    } else {
        lex_with_offset(source, base_offset)
    }
}

fn lex_with_offset(source: &str, base_offset: usize) -> LexOutput {
    let mut lexer = RawTokenKind::lexer(source);
    let mut tokens = Vec::with_capacity(source.len() / 3 + 8);
    let mut diagnostics = Vec::new();

    while let Some(raw) = lexer.next() {
        let raw_span = lexer.span();
        let span = Span::new(base_offset + raw_span.start, base_offset + raw_span.end);

        match raw {
            Ok(RawTokenKind::BlockComment) => {
                // The reference Wren lexer treats the entire nested block comment
                // as whitespace, including any newlines contained in it.
            }
            Ok(kind) => {
                if kind == RawTokenKind::Hash && source[raw_span.start..].starts_with("#!/") {
                    diagnostics.push(Diagnostic::error(
                        "WREN-L008",
                        "a shebang is only allowed at the start of the source",
                        span,
                    ));
                }
                if let Some(kind) = map_raw_token(kind) {
                    tokens.push(Token::new(kind, span));
                }
            }
            Err(error) => {
                let (code, message) = match error {
                    LexErrorKind::UnterminatedBlockComment => {
                        ("WREN-L001", "unterminated block comment")
                    }
                    LexErrorKind::UnterminatedString => ("WREN-L002", "unterminated string literal"),
                    LexErrorKind::UnterminatedRawString => {
                        ("WREN-L003", "unterminated raw string literal")
                    }
                    LexErrorKind::InterpolationTooDeep => {
                        ("WREN-L004", "string interpolation is nested too deeply")
                    }
                    LexErrorKind::UnexpectedCharacter => {
                        ("WREN-L005", "unexpected character in source")
                    }
                    LexErrorKind::InvalidInterpolation => {
                        ("WREN-L006", "expected `(` after `%` in string interpolation")
                    }
                    LexErrorKind::UnterminatedScientificNotation => {
                        ("WREN-L007", "unterminated scientific notation")
                    }
                };
                diagnostics.push(
                    Diagnostic::error(code, message, span)
                        .with_primary_label(span, "the tokenizer could not consume this input"),
                );
            }
        }
    }

    let eof = base_offset + source.len();
    tokens.push(Token::new(TokenKind::Eof, Span::empty(eof)));
    LexOutput {
        tokens,
        diagnostics,
    }
}

fn map_raw_token(raw: RawTokenKind) -> Option<TokenKind> {
    use RawTokenKind as R;
    use TokenKind as T;
    Some(match raw {
        R::_Whitespace | R::_LineComment | R::BlockComment => return None,
        R::Newline => T::Newline,
        R::As => T::As,
        R::Break => T::Break,
        R::Class => T::Class,
        R::Construct => T::Construct,
        R::Continue => T::Continue,
        R::Else => T::Else,
        R::False => T::False,
        R::For => T::For,
        R::Foreign => T::Foreign,
        R::If => T::If,
        R::Import => T::Import,
        R::In => T::In,
        R::Is => T::Is,
        R::Null => T::Null,
        R::Return => T::Return,
        R::Static => T::Static,
        R::Super => T::Super,
        R::This => T::This,
        R::True => T::True,
        R::Var => T::Var,
        R::While => T::While,
        R::RawString => T::RawString,
        R::String => T::String,
        R::Number | R::NumberDec => T::Number,
        R::Field => T::Field,
        R::StaticField => T::StaticField,
        R::Identifier => T::Identifier,
        R::DotDotDot => T::DotDotDot,
        R::DotDot => T::DotDot,
        R::EqualEqual => T::EqualEqual,
        R::BangEqual => T::BangEqual,
        R::LessEqual => T::LessEqual,
        R::GreaterEqual => T::GreaterEqual,
        R::ShiftLeft => T::ShiftLeft,
        R::ShiftRight => T::ShiftRight,
        R::AndAnd => T::AndAnd,
        R::OrOr => T::OrOr,
        R::LeftParen => T::LeftParen,
        R::RightParen => T::RightParen,
        R::LeftBracket => T::LeftBracket,
        R::RightBracket => T::RightBracket,
        R::LeftBrace => T::LeftBrace,
        R::RightBrace => T::RightBrace,
        R::Comma => T::Comma,
        R::Dot => T::Dot,
        R::Colon => T::Colon,
        R::Question => T::Question,
        R::Equal => T::Equal,
        R::Plus => T::Plus,
        R::Minus => T::Minus,
        R::Star => T::Star,
        R::Slash => T::Slash,
        R::Percent => T::Percent,
        R::Bang => T::Bang,
        R::Tilde => T::Tilde,
        R::Ampersand => T::Ampersand,
        R::Pipe => T::Pipe,
        R::Caret => T::Caret,
        R::Less => T::Less,
        R::Greater => T::Greater,
        R::Hash => T::Hash,
    })
}

fn lex_decimal_number(
    lexer: &mut logos::Lexer<'_, RawTokenKind>,
) -> Result<(), LexErrorKind> {
    let bytes = lexer.slice().as_bytes();
    if let Some(exponent) = bytes.iter().position(|&byte| matches!(byte, b'e' | b'E')) {
        let mut index = exponent + 1;
        if matches!(bytes.get(index).copied(), Some(b'+') | Some(b'-')) {
            index += 1;
        }
        if index == bytes.len() {
            return Err(LexErrorKind::UnterminatedScientificNotation);
        }
    }
    Ok(())
}

fn lex_block_comment(lexer: &mut logos::Lexer<'_, RawTokenKind>) -> Result<(), LexErrorKind> {
    let bytes = lexer.remainder().as_bytes();
    let mut index = 0;
    let mut depth = 1usize;

    while index + 1 < bytes.len() {
        match (bytes[index], bytes[index + 1]) {
            (b'/', b'*') => {
                depth += 1;
                index += 2;
            }
            (b'*', b'/') => {
                depth -= 1;
                index += 2;
                if depth == 0 {
                    lexer.bump(index);
                    return Ok(());
                }
            }
            _ => index += 1,
        }
    }

    lexer.bump(bytes.len());
    Err(LexErrorKind::UnterminatedBlockComment)
}

fn lex_raw_string(lexer: &mut logos::Lexer<'_, RawTokenKind>) -> Result<(), LexErrorKind> {
    let bytes = lexer.remainder().as_bytes();
    if let Some(consumed) = scan_raw_string_tail(bytes) {
        lexer.bump(consumed);
        Ok(())
    } else {
        lexer.bump(bytes.len());
        Err(LexErrorKind::UnterminatedRawString)
    }
}

fn lex_normal_string(lexer: &mut logos::Lexer<'_, RawTokenKind>) -> Result<(), LexErrorKind> {
    let bytes = lexer.remainder().as_bytes();
    match scan_normal_string_tail(bytes, 0) {
        Ok(consumed) => {
            lexer.bump(consumed);
            Ok(())
        }
        Err(error) => {
            lexer.bump(bytes.len());
            Err(error)
        }
    }
}

fn scan_raw_string_tail(bytes: &[u8]) -> Option<usize> {
    let mut index = 0;
    while index + 2 < bytes.len() {
        if bytes[index] == b'"' && bytes[index + 1] == b'"' && bytes[index + 2] == b'"' {
            return Some(index + 3);
        }
        index += 1;
    }
    None
}

/// Scans the bytes after an opening normal quote and returns the number of bytes
/// consumed through the matching closing quote.
fn scan_normal_string_tail(
    bytes: &[u8],
    interpolation_depth: usize,
) -> Result<usize, LexErrorKind> {
    if interpolation_depth > MAX_INTERPOLATION_DEPTH {
        return Err(LexErrorKind::InterpolationTooDeep);
    }

    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => {
                index += 1;
                if index < bytes.len() {
                    index += 1;
                }
            }
            b'"' => return Ok(index + 1),
            b'%' => {
                if index + 1 >= bytes.len() || bytes[index + 1] != b'(' {
                    return Err(LexErrorKind::InvalidInterpolation);
                }
                let body_start = index + 2;
                let consumed =
                    scan_interpolation_tail(&bytes[body_start..], interpolation_depth + 1)?;
                index = body_start + consumed;
            }
            _ => index += 1,
        }
    }

    Err(LexErrorKind::UnterminatedString)
}

/// Scans bytes after `%(` and returns the number of bytes consumed through its
/// matching `)`. Nested strings, comments and parentheses are handled.
fn scan_interpolation_tail(bytes: &[u8], depth: usize) -> Result<usize, LexErrorKind> {
    if depth > MAX_INTERPOLATION_DEPTH {
        return Err(LexErrorKind::InterpolationTooDeep);
    }

    let mut index = 0;
    let mut parens = 1usize;

    while index < bytes.len() {
        if index + 2 < bytes.len()
            && bytes[index] == b'"'
            && bytes[index + 1] == b'"'
            && bytes[index + 2] == b'"'
        {
            let tail = &bytes[index + 3..];
            let Some(consumed) = scan_raw_string_tail(tail) else {
                return Err(LexErrorKind::UnterminatedRawString);
            };
            index += 3 + consumed;
            continue;
        }

        if index + 1 < bytes.len() && bytes[index] == b'/' && bytes[index + 1] == b'/' {
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' && bytes[index] != b'\r' {
                index += 1;
            }
            continue;
        }

        if index + 1 < bytes.len() && bytes[index] == b'/' && bytes[index + 1] == b'*' {
            index += 2;
            let mut comment_depth = 1usize;
            while index + 1 < bytes.len() {
                match (bytes[index], bytes[index + 1]) {
                    (b'/', b'*') => {
                        comment_depth += 1;
                        index += 2;
                    }
                    (b'*', b'/') => {
                        comment_depth -= 1;
                        index += 2;
                        if comment_depth == 0 {
                            break;
                        }
                    }
                    _ => index += 1,
                }
            }
            if comment_depth != 0 {
                return Err(LexErrorKind::UnterminatedBlockComment);
            }
            continue;
        }

        match bytes[index] {
            b'"' => {
                let consumed = scan_normal_string_tail(&bytes[index + 1..], depth)?;
                index += 1 + consumed;
            }
            b'(' => {
                parens += 1;
                index += 1;
            }
            b')' => {
                parens -= 1;
                index += 1;
                if parens == 0 {
                    return Ok(index);
                }
            }
            _ => index += 1,
        }
    }

    Err(LexErrorKind::UnterminatedString)
}

// =============================================================================
// AST
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct ModuleAst {
    pub statements: Vec<Stmt>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StmtKind {
    Variable {
        name: Ident,
        initializer: Option<Expr>,
    },
    Class(ClassDecl),
    Import(ImportDecl),
    Block(Vec<Stmt>),
    If {
        condition: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },
    While {
        condition: Expr,
        body: Box<Stmt>,
    },
    For {
        name: Ident,
        iterable: Expr,
        body: Box<Stmt>,
    },
    Break,
    Continue,
    Return(Option<Expr>),
    Expression(Expr),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportDecl {
    pub module: Vec<u8>,
    pub bindings: Vec<ImportBinding>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportBinding {
    pub imported: Ident,
    pub local: Ident,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClassDecl {
    pub name: Ident,
    pub superclass: Option<Expr>,
    pub is_foreign: bool,
    pub attributes: Vec<Attribute>,
    pub methods: Vec<MethodDecl>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MethodDecl {
    pub signature: MethodSignatureAst,
    pub is_static: bool,
    pub is_foreign: bool,
    pub is_constructor: bool,
    pub attributes: Vec<Attribute>,
    pub body: Option<FunctionBody>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MethodSignatureAst {
    Getter {
        name: Ident,
    },
    Method {
        name: Ident,
        params: Vec<Ident>,
    },
    Setter {
        name: Ident,
        param: Ident,
    },
    PrefixOperator {
        operator: OperatorName,
    },
    InfixOperator {
        operator: OperatorName,
        param: Ident,
    },
    SubscriptGetter {
        params: Vec<Ident>,
    },
    SubscriptSetter {
        params: Vec<Ident>,
        value: Ident,
    },
}

impl MethodSignatureAst {
    pub fn explicit_arity(&self) -> usize {
        match self {
            MethodSignatureAst::Getter { .. } | MethodSignatureAst::PrefixOperator { .. } => 0,
            MethodSignatureAst::Method { params, .. }
            | MethodSignatureAst::SubscriptGetter { params } => params.len(),
            MethodSignatureAst::Setter { .. } | MethodSignatureAst::InfixOperator { .. } => 1,
            MethodSignatureAst::SubscriptSetter { params, .. } => params.len() + 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperatorName {
    Bang,
    Tilde,
    Minus,
    Plus,
    Star,
    Slash,
    Percent,
    DotDot,
    DotDotDot,
    ShiftLeft,
    ShiftRight,
    Ampersand,
    Caret,
    Pipe,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    EqualEqual,
    BangEqual,
}

impl OperatorName {
    pub const fn as_str(self) -> &'static str {
        match self {
            OperatorName::Bang => "!",
            OperatorName::Tilde => "~",
            OperatorName::Minus => "-",
            OperatorName::Plus => "+",
            OperatorName::Star => "*",
            OperatorName::Slash => "/",
            OperatorName::Percent => "%",
            OperatorName::DotDot => "..",
            OperatorName::DotDotDot => "...",
            OperatorName::ShiftLeft => "<<",
            OperatorName::ShiftRight => ">>",
            OperatorName::Ampersand => "&",
            OperatorName::Caret => "^",
            OperatorName::Pipe => "|",
            OperatorName::Less => "<",
            OperatorName::LessEqual => "<=",
            OperatorName::Greater => ">",
            OperatorName::GreaterEqual => ">=",
            OperatorName::EqualEqual => "==",
            OperatorName::BangEqual => "!=",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionBody {
    pub params: Vec<Ident>,
    pub kind: FunctionBodyKind,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FunctionBodyKind {
    /// The Wren single-expression block form. The expression is implicitly returned.
    Expression(Box<Expr>),
    /// The normal newline-led block form. Falling off the end returns null.
    Statements(Vec<Stmt>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Attribute {
    pub runtime: bool,
    pub group: Option<Ident>,
    pub entries: Vec<AttributeEntry>,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttributeEntry {
    pub key: Ident,
    pub value: AttributeValue,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AttributeValue {
    Null,
    Name(String),
    String(Vec<u8>),
    Bool(bool),
    Number(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

impl Expr {
    #[inline]
    pub fn new(kind: ExprKind, span: Span) -> Self {
        Self { kind, span }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Null,
    Bool(bool),
    Number(f64),
    String(Vec<u8>),
    InterpolatedString(Vec<StringPart>),
    Name(Ident),
    Field(Ident),
    StaticField(Ident),
    This,
    Super(SuperExpr),
    List(Vec<Expr>),
    Map(Vec<MapEntry>),
    Grouped(Box<Expr>),
    Unary {
        operator: UnaryOp,
        operand: Box<Expr>,
    },
    Binary {
        left: Box<Expr>,
        operator: BinaryOp,
        right: Box<Expr>,
    },
    Logical {
        left: Box<Expr>,
        operator: LogicalOp,
        right: Box<Expr>,
    },
    Conditional {
        condition: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
    },
    Invoke(InvokeExpr),
    Subscript {
        receiver: Box<Expr>,
        args: Vec<Expr>,
    },
    Closure(FunctionBody),
}

#[derive(Clone, Debug, PartialEq)]
pub struct InvokeExpr {
    /// `None` means an implicit self-send such as `foo()`.
    pub receiver: Option<Box<Expr>>,
    pub name: Ident,
    pub args: Vec<Expr>,
    pub syntax: InvokeSyntax,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InvokeSyntax {
    Getter,
    Method,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SuperExpr {
    /// `None` means "use the enclosing method name", matching Wren's unnamed
    /// `super` call syntax. Parentheses still determine getter vs. method call.
    pub method: Option<Ident>,
    pub args: Vec<Expr>,
    pub syntax: InvokeSyntax,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MapEntry {
    pub key: Expr,
    pub value: Expr,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq)]
pub enum StringPart {
    Bytes(Vec<u8>),
    Expression(Expr),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Negate,
    Not,
    Complement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Multiply,
    Divide,
    Modulo,
    Add,
    Subtract,
    InclusiveRange,
    ExclusiveRange,
    ShiftLeft,
    ShiftRight,
    BitAnd,
    BitXor,
    BitOr,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Is,
    Equal,
    NotEqual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LogicalOp {
    And,
    Or,
}

// =============================================================================
// Public parser API
// =============================================================================

#[derive(Debug)]
pub struct ParseOutput {
    pub module: ModuleAst,
    pub diagnostics: Vec<Diagnostic>,
}

impl ParseOutput {
    #[inline]
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
    }

    pub fn emit(&self, source_name: &str, source: &str) -> io::Result<()> {
        emit_diagnostics(source_name, source, &self.diagnostics)
    }
}

pub fn parse(source_name: &str, source: &str) -> ParseOutput {
    let lexed = lex(source);
    let mut parser = Parser::new(source_name, source, lexed.tokens);
    parser.diagnostics.extend(lexed.diagnostics);
    let module = parser.parse_module();
    ParseOutput {
        module,
        diagnostics: parser.diagnostics,
    }
}

/// Convenience entry point for CLI-style use.
///
/// Diagnostics are always emitted through Ariadne. The AST is returned only if
/// no error diagnostic was produced.
pub fn parse_and_report(source_name: &str, source: &str) -> Result<ModuleAst, ()> {
    let output = parse(source_name, source);
    let _ = output.emit(source_name, source);
    if output.has_errors() {
        Err(())
    } else {
        Ok(output.module)
    }
}

// =============================================================================
// Recursive-descent / Pratt parser
// =============================================================================

type PResult<T> = Result<T, ()>;

struct Parser<'a> {
    source_name: &'a str,
    source: &'a str,
    tokens: Vec<Token>,
    current: usize,
    diagnostics: Vec<Diagnostic>,
    recursion_depth: usize,
}

impl<'a> Parser<'a> {
    fn new(source_name: &'a str, source: &'a str, tokens: Vec<Token>) -> Self {
        Self {
            source_name,
            source,
            tokens,
            current: 0,
            diagnostics: Vec::new(),
            recursion_depth: 0,
        }
    }

    fn parse_module(&mut self) -> ModuleAst {
        let start = self.peek().span.start;
        let mut statements = Vec::new();
        self.skip_newlines();

        while !self.at(TokenKind::Eof) {
            let before = self.current;
            match self.parse_declaration() {
                Ok(statement) => {
                    statements.push(statement);
                    self.consume_statement_boundary();
                    self.skip_newlines();
                }
                Err(()) => self.synchronize_statement(),
            }
            // A stray '}' belongs to no enclosing block here. Recovery normally
            // leaves it for the caller, but at module scope we must consume it.
            if self.current == before {
                self.advance();
            }
        }

        ModuleAst {
            statements,
            span: Span::new(start, self.peek().span.end),
        }
    }

    fn parse_declaration(&mut self) -> PResult<Stmt> {
        if self.at(TokenKind::Hash) {
            let attributes = self.parse_attributes()?;
            let start = attributes
                .first()
                .map(|attribute| attribute.span.start)
                .unwrap_or(self.peek().span.start);
            let is_foreign = self.match_kind(TokenKind::Foreign);
            if !self.match_kind(TokenKind::Class) {
                return self.error_at_current(
                    "WREN-P001",
                    "attributes at statement scope must annotate a class",
                    "expected `class` after attribute list",
                );
            }
            return self.parse_class_after_keyword(start, is_foreign, attributes);
        }

        if self.match_kind(TokenKind::Foreign) {
            let start = self.previous().span.start;
            self.expect(
                TokenKind::Class,
                "WREN-P002",
                "expected `class` after `foreign`",
            )?;
            return self.parse_class_after_keyword(start, true, Vec::new());
        }

        if self.match_kind(TokenKind::Class) {
            let start = self.previous().span.start;
            return self.parse_class_after_keyword(start, false, Vec::new());
        }

        if self.match_kind(TokenKind::Import) {
            return self.parse_import(self.previous().span.start);
        }

        if self.match_kind(TokenKind::Var) {
            return self.parse_variable(self.previous().span.start);
        }

        self.parse_statement()
    }

    fn parse_variable(&mut self, start: usize) -> PResult<Stmt> {
        let name = self.parse_identifier("WREN-P003", "expected variable name after `var`")?;
        let initializer = if self.match_kind(TokenKind::Equal) {
            self.skip_newlines();
            Some(self.parse_expression()?)
        } else {
            None
        };
        let end = initializer.as_ref().map_or(name.span.end, |expr| expr.span.end);
        Ok(Stmt {
            kind: StmtKind::Variable { name, initializer },
            span: Span::new(start, end),
        })
    }

    fn parse_import(&mut self, start: usize) -> PResult<Stmt> {
        self.skip_newlines();
        let module_token = self.peek();
        if !matches!(module_token.kind, TokenKind::String | TokenKind::RawString) {
            return self.error_at_current(
                "WREN-P004",
                "expected a module string after `import`",
                "module names must be string literals",
            );
        }
        self.advance();

        let (module, interpolated) = self.decode_string_for_import(module_token)?;
        if interpolated {
            self.diagnostics.push(
                Diagnostic::error(
                    "WREN-P005",
                    "module names cannot use string interpolation",
                    module_token.span,
                )
                .with_primary_label(module_token.span, "use a plain string literal here"),
            );
        }

        let mut bindings = Vec::new();
        let mut end = module_token.span.end;
        if self.match_kind(TokenKind::For) {
            loop {
                self.skip_newlines();
                let imported = self.parse_identifier(
                    "WREN-P006",
                    "expected imported variable name after `for`",
                )?;
                let local = if self.match_kind(TokenKind::As) {
                    self.parse_identifier("WREN-P007", "expected local name after `as`")?
                } else {
                    imported.clone()
                };
                let span = imported.span.join(local.span);
                end = span.end;
                bindings.push(ImportBinding {
                    imported,
                    local,
                    span,
                });

                if !self.match_kind(TokenKind::Comma) {
                    break;
                }
            }
        }

        Ok(Stmt {
            kind: StmtKind::Import(ImportDecl { module, bindings }),
            span: Span::new(start, end),
        })
    }

    fn parse_class_after_keyword(
        &mut self,
        start: usize,
        is_foreign: bool,
        attributes: Vec<Attribute>,
    ) -> PResult<Stmt> {
        let name = self.parse_identifier("WREN-P008", "expected class name")?;

        let superclass = if self.match_kind(TokenKind::Is) {
            if self.at(TokenKind::Newline) {
                return self.error_at_current(
                    "WREN-P071",
                    "expected superclass expression after `is`",
                    "a class header cannot break the line immediately after `is`",
                );
            }
            Some(self.parse_precedence(16)?)
        } else {
            None
        };

        self.expect(
            TokenKind::LeftBrace,
            "WREN-P010",
            "expected `{` before class body",
        )?;
        self.skip_newlines();

        let mut methods = Vec::new();
        while !self.at(TokenKind::RightBrace) && !self.at(TokenKind::Eof) {
            let method_attributes = if self.at(TokenKind::Hash) {
                self.parse_attributes()?
            } else {
                Vec::new()
            };
            self.skip_newlines();

            match self.parse_method(method_attributes) {
                Ok(method) => {
                    methods.push(method);
                    self.consume_statement_boundary();
                    self.skip_newlines();
                }
                Err(()) => self.synchronize_class_member(),
            }
        }

        let close = self.expect(
            TokenKind::RightBrace,
            "WREN-P011",
            "expected `}` after class body",
        )?;

        Ok(Stmt {
            kind: StmtKind::Class(ClassDecl {
                name,
                superclass,
                is_foreign,
                attributes,
                methods,
            }),
            span: Span::new(start, close.span.end),
        })
    }

    fn parse_method(&mut self, attributes: Vec<Attribute>) -> PResult<MethodDecl> {
        let start = attributes
            .first()
            .map(|attribute| attribute.span.start)
            .unwrap_or(self.peek().span.start);

        // Wren's modifier order is fixed: `foreign` may precede `static`.
        let is_foreign = self.match_kind(TokenKind::Foreign);
        let is_static = self.match_kind(TokenKind::Static);

        let is_constructor = self.match_kind(TokenKind::Construct);
        if is_constructor && is_static {
            self.diagnostics.push(
                Diagnostic::error(
                    "WREN-P012",
                    "constructors cannot be static",
                    self.previous().span,
                )
                .with_primary_label(self.previous().span, "remove `static`"),
            );
        }

        let signature = if is_constructor {
            self.parse_constructor_signature()?
        } else {
            self.parse_method_signature()?
        };

        if signature.explicit_arity() > MAX_PARAMETERS {
            self.diagnostics.push(
                Diagnostic::error(
                    "WREN-P013",
                    format!(
                        "method has {} parameters, but the VM supports at most {}",
                        signature.explicit_arity(),
                        MAX_PARAMETERS
                    ),
                    Span::new(start, self.previous().span.end),
                )
                .with_note("The bytecode ABI has ARG0..ARG15."),
            );
        }

        let body = if is_foreign {
            None
        } else {
            Some(self.parse_function_body(false)?)
        };

        let end = body
            .as_ref()
            .map_or(self.previous().span.end, |body| body.span.end);

        Ok(MethodDecl {
            signature,
            is_static,
            is_foreign,
            is_constructor,
            attributes,
            body,
            span: Span::new(start, end),
        })
    }

    fn parse_constructor_signature(&mut self) -> PResult<MethodSignatureAst> {
        let name = self.parse_identifier("WREN-P014", "expected constructor name")?;
        self.expect(
            TokenKind::LeftParen,
            "WREN-P015",
            "constructors require a parenthesized parameter list",
        )?;

        // Constructors permit an immediately empty `()`. If parameters are
        // present, Wren permits newlines before the first parameter but not
        // between the final parameter and `)`.
        let params = if self.match_kind(TokenKind::RightParen) {
            Vec::new()
        } else {
            self.parse_nonempty_parameter_list(TokenKind::RightParen)?
        };
        Ok(MethodSignatureAst::Method { name, params })
    }

    fn parse_method_signature(&mut self) -> PResult<MethodSignatureAst> {
        if self.match_kind(TokenKind::LeftBracket) {
            if self.at(TokenKind::RightBracket) {
                let span = self.peek().span;
                self.diagnostics.push(
                    Diagnostic::error(
                        "WREN-P016",
                        "a subscript method needs at least one parameter",
                        span,
                    )
                    .with_primary_label(span, "empty subscript signature"),
                );
                self.advance();

                // Keep parsing an invalid empty subscript setter so recovery does
                // not leave `=(value)` behind as a cascade of unrelated errors.
                if self.match_kind(TokenKind::Equal) {
                    self.expect(
                        TokenKind::LeftParen,
                        "WREN-P017",
                        "expected `(value)` after subscript setter `=`",
                    )?;
                    let value = self.parse_identifier(
                        "WREN-P018",
                        "expected value parameter in subscript setter",
                    )?;
                    self.expect(
                        TokenKind::RightParen,
                        "WREN-P019",
                        "expected `)` after subscript setter value",
                    )?;
                    return Ok(MethodSignatureAst::SubscriptSetter {
                        params: Vec::new(),
                        value,
                    });
                }

                return Ok(MethodSignatureAst::SubscriptGetter { params: Vec::new() });
            }
            let params = self.parse_nonempty_parameter_list(TokenKind::RightBracket)?;
            if self.match_kind(TokenKind::Equal) {
                self.expect(
                    TokenKind::LeftParen,
                    "WREN-P017",
                    "expected `(value)` after subscript setter `=`",
                )?;
                let value = self.parse_identifier(
                    "WREN-P018",
                    "expected value parameter in subscript setter",
                )?;
                self.expect(
                    TokenKind::RightParen,
                    "WREN-P019",
                    "expected `)` after subscript setter value",
                )?;
                return Ok(MethodSignatureAst::SubscriptSetter { params, value });
            }
            return Ok(MethodSignatureAst::SubscriptGetter { params });
        }

        if let Some(operator) = self.peek_operator_name() {
            let operator_span = self.advance().span;

            if matches!(operator, OperatorName::Bang | OperatorName::Tilde) {
                if self.at(TokenKind::LeftParen) {
                    self.diagnostics.push(
                        Diagnostic::error(
                            "WREN-P020",
                            format!("operator `{}` is unary in Wren", operator.as_str()),
                            operator_span,
                        )
                        .with_primary_label(operator_span, "remove the parameter list"),
                    );
                }
                return Ok(MethodSignatureAst::PrefixOperator { operator });
            }

            if operator == OperatorName::Minus && !self.at(TokenKind::LeftParen) {
                return Ok(MethodSignatureAst::PrefixOperator { operator });
            }

            self.expect(
                TokenKind::LeftParen,
                "WREN-P020",
                format!("operator `{}` requires one parameter", operator.as_str()),
            )?;
            let param = self.parse_identifier(
                "WREN-P021",
                "expected right operand parameter",
            )?;
            self.expect(
                TokenKind::RightParen,
                "WREN-P022",
                "expected `)` after operator parameter",
            )?;
            return Ok(MethodSignatureAst::InfixOperator { operator, param });
        }

        let name = self.parse_identifier("WREN-P023", "expected method name")?;
        if self.match_kind(TokenKind::Equal) {
            self.expect(
                TokenKind::LeftParen,
                "WREN-P024",
                "expected `(value)` after setter `=`",
            )?;
            let param = self.parse_identifier("WREN-P025", "expected setter value parameter")?;
            self.expect(
                TokenKind::RightParen,
                "WREN-P026",
                "expected `)` after setter parameter",
            )?;
            return Ok(MethodSignatureAst::Setter { name, param });
        }

        if self.match_kind(TokenKind::LeftParen) {
            // Named methods allow newlines before an empty argument list.
            self.skip_newlines();
            let params = if self.match_kind(TokenKind::RightParen) {
                Vec::new()
            } else {
                self.parse_nonempty_parameter_list(TokenKind::RightParen)?
            };
            return Ok(MethodSignatureAst::Method { name, params });
        }

        Ok(MethodSignatureAst::Getter { name })
    }

    fn parse_nonempty_parameter_list(&mut self, close: TokenKind) -> PResult<Vec<Ident>> {
        let mut params = Vec::new();

        loop {
            // This mirrors Wren's finishParameterList(): a newline is allowed
            // before each parameter, but the comma/closing delimiter must
            // immediately follow the parameter itself.
            self.skip_newlines();
            params.push(self.parse_identifier("WREN-P027", "expected parameter name")?);
            if self.match_kind(close) {
                break;
            }
            self.expect(
                TokenKind::Comma,
                "WREN-P028",
                format!("expected `,` or {} in parameter list", close.description()),
            )?;
        }
        Ok(params)
    }

    fn parse_statement(&mut self) -> PResult<Stmt> {
        if self.match_kind(TokenKind::If) {
            return self.parse_if(self.previous().span.start);
        }
        if self.match_kind(TokenKind::While) {
            return self.parse_while(self.previous().span.start);
        }
        if self.match_kind(TokenKind::For) {
            return self.parse_for(self.previous().span.start);
        }
        if self.match_kind(TokenKind::Break) {
            let span = self.previous().span;
            return Ok(Stmt {
                kind: StmtKind::Break,
                span,
            });
        }
        if self.match_kind(TokenKind::Continue) {
            let span = self.previous().span;
            return Ok(Stmt {
                kind: StmtKind::Continue,
                span,
            });
        }
        if self.match_kind(TokenKind::Return) {
            return self.parse_return(self.previous().span.start);
        }
        if self.match_kind(TokenKind::LeftBrace) {
            return self.parse_block_statement(self.previous().span.start);
        }
        let expression = self.parse_expression()?;
        let span = expression.span;
        Ok(Stmt {
            kind: StmtKind::Expression(expression),
            span,
        })
    }

    fn parse_if(&mut self, start: usize) -> PResult<Stmt> {
        self.expect(
            TokenKind::LeftParen,
            "WREN-P029",
            "expected `(` after `if`",
        )?;
        let condition = self.parse_expression()?;
        self.expect(
            TokenKind::RightParen,
            "WREN-P030",
            "expected `)` after if condition",
        )?;
        let then_branch = Box::new(self.parse_statement()?);

        let else_branch = if self.match_kind(TokenKind::Else) {
            Some(Box::new(self.parse_statement()?))
        } else {
            None
        };

        let end = else_branch
            .as_ref()
            .map_or(then_branch.span.end, |branch| branch.span.end);
        Ok(Stmt {
            kind: StmtKind::If {
                condition,
                then_branch,
                else_branch,
            },
            span: Span::new(start, end),
        })
    }

    fn parse_while(&mut self, start: usize) -> PResult<Stmt> {
        self.expect(
            TokenKind::LeftParen,
            "WREN-P031",
            "expected `(` after `while`",
        )?;
        let condition = self.parse_expression()?;
        self.expect(
            TokenKind::RightParen,
            "WREN-P032",
            "expected `)` after while condition",
        )?;
        let body = Box::new(self.parse_statement()?);
        let end = body.span.end;
        Ok(Stmt {
            kind: StmtKind::While { condition, body },
            span: Span::new(start, end),
        })
    }

    fn parse_for(&mut self, start: usize) -> PResult<Stmt> {
        self.expect(
            TokenKind::LeftParen,
            "WREN-P033",
            "expected `(` after `for`",
        )?;
        let name = self.parse_identifier("WREN-P034", "expected loop variable name")?;
        self.expect(TokenKind::In, "WREN-P035", "expected `in` in for loop")?;
        self.skip_newlines();
        let iterable = self.parse_expression()?;
        self.expect(
            TokenKind::RightParen,
            "WREN-P036",
            "expected `)` after for sequence",
        )?;
        let body = Box::new(self.parse_statement()?);
        let end = body.span.end;
        Ok(Stmt {
            kind: StmtKind::For {
                name,
                iterable,
                body,
            },
            span: Span::new(start, end),
        })
    }

    fn parse_return(&mut self, start: usize) -> PResult<Stmt> {
        if self.at(TokenKind::Newline)
            || self.at(TokenKind::RightBrace)
            || self.at(TokenKind::Eof)
        {
            return Ok(Stmt {
                kind: StmtKind::Return(None),
                span: Span::new(start, self.previous().span.end),
            });
        }

        let value = self.parse_expression()?;
        let end = value.span.end;
        Ok(Stmt {
            kind: StmtKind::Return(Some(value)),
            span: Span::new(start, end),
        })
    }

    fn parse_block_statement(&mut self, start: usize) -> PResult<Stmt> {
        if self.match_kind(TokenKind::RightBrace) {
            return Ok(Stmt {
                kind: StmtKind::Block(Vec::new()),
                span: Span::new(start, self.previous().span.end),
            });
        }

        if !self.at(TokenKind::Newline) {
            let expression = self.parse_expression()?;
            let expression_span = expression.span;
            let close = self.expect(
                TokenKind::RightBrace,
                "WREN-P037",
                "single-line blocks may contain only one expression",
            )?;
            return Ok(Stmt {
                kind: StmtKind::Block(vec![Stmt {
                    kind: StmtKind::Expression(expression),
                    span: expression_span,
                }]),
                span: Span::new(start, close.span.end),
            });
        }

        self.skip_newlines();
        if self.match_kind(TokenKind::RightBrace) {
            return Ok(Stmt {
                kind: StmtKind::Block(Vec::new()),
                span: Span::new(start, self.previous().span.end),
            });
        }

        let mut statements = Vec::new();
        while !self.at(TokenKind::RightBrace) && !self.at(TokenKind::Eof) {
            match self.parse_declaration() {
                Ok(statement) => {
                    statements.push(statement);
                    self.consume_required_line(
                        "WREN-P037",
                        "expected newline after statement in block",
                    );
                }
                Err(()) => self.synchronize_statement(),
            }
        }
        let close = self.expect(
            TokenKind::RightBrace,
            "WREN-P037",
            "expected `}` after block",
        )?;
        Ok(Stmt {
            kind: StmtKind::Block(statements),
            span: Span::new(start, close.span.end),
        })
    }

    fn parse_function_body(&mut self, allow_params: bool) -> PResult<FunctionBody> {
        let open = self.expect(
            TokenKind::LeftBrace,
            "WREN-P038",
            "expected `{` to start function body",
        )?;
        self.parse_function_body_after_open(open.span.start, allow_params)
    }

    fn parse_function_body_after_open(
        &mut self,
        start: usize,
        allow_params: bool,
    ) -> PResult<FunctionBody> {
        let mut params = Vec::new();
        if allow_params && self.match_kind(TokenKind::Pipe) {
            loop {
                self.skip_newlines();
                params.push(self.parse_identifier(
                    "WREN-P039",
                    "expected closure parameter name",
                )?);
                if self.match_kind(TokenKind::Pipe) {
                    break;
                }
                self.expect(
                    TokenKind::Comma,
                    "WREN-P040",
                    "expected `,` or `|` in closure parameter list",
                )?;
                self.skip_newlines();
            }
            if params.len() > MAX_PARAMETERS {
                self.diagnostics.push(
                    Diagnostic::error(
                        "WREN-P041",
                        format!(
                            "closure has {} parameters, maximum is {}",
                            params.len(),
                            MAX_PARAMETERS
                        ),
                        Span::new(start, self.previous().span.end),
                    )
                    .with_note("The bytecode ABI has ARG0..ARG15."),
                );
            }
        }

        if self.match_kind(TokenKind::RightBrace) {
            let end = self.previous().span.end;
            return Ok(FunctionBody {
                params,
                kind: FunctionBodyKind::Statements(Vec::new()),
                span: Span::new(start, end),
            });
        }

        if self.at(TokenKind::Newline) {
            self.skip_newlines();
            let mut statements = Vec::new();
            while !self.at(TokenKind::RightBrace) && !self.at(TokenKind::Eof) {
                match self.parse_declaration() {
                    Ok(statement) => {
                        statements.push(statement);
                        self.consume_required_line(
                            "WREN-P042",
                            "expected newline after statement in function body",
                        );
                    }
                    Err(()) => self.synchronize_statement(),
                }
            }
            let close = self.expect(
                TokenKind::RightBrace,
                "WREN-P042",
                "expected `}` after function body",
            )?;
            return Ok(FunctionBody {
                params,
                kind: FunctionBodyKind::Statements(statements),
                span: Span::new(start, close.span.end),
            });
        }

        let expression = self.parse_expression()?;
        let close = self.expect(
            TokenKind::RightBrace,
            "WREN-P043",
            "single-line function bodies may contain only one expression",
        )?;
        Ok(FunctionBody {
            params,
            kind: FunctionBodyKind::Expression(Box::new(expression)),
            span: Span::new(start, close.span.end),
        })
    }

    fn parse_attributes(&mut self) -> PResult<Vec<Attribute>> {
        let mut attributes = Vec::new();
        while self.match_kind(TokenKind::Hash) {
            let start = self.previous().span.start;
            let runtime = self.match_kind(TokenKind::Bang);
            let key_or_group = self.parse_identifier(
                "WREN-P044",
                "expected attribute name after `#`",
            )?;

            if self.match_kind(TokenKind::LeftParen) {
                let group = key_or_group;
                let mut entries = Vec::new();
                self.skip_newlines();
                if self.match_kind(TokenKind::RightParen) {
                    self.diagnostics.push(
                        Diagnostic::error(
                            "WREN-P045",
                            "attribute groups cannot be empty",
                            Span::new(group.span.start, self.previous().span.end),
                        )
                        .with_primary_label(
                            Span::new(group.span.start, self.previous().span.end),
                            "add at least one attribute to this group",
                        ),
                    );
                } else {
                    loop {
                        self.skip_newlines();
                        let key = self.parse_identifier(
                            "WREN-P045",
                            "expected attribute key",
                        )?;
                        let value = if self.match_kind(TokenKind::Equal) {
                            self.parse_attribute_value()?
                        } else {
                            AttributeValue::Null
                        };
                        let entry_span = Span::new(key.span.start, self.previous().span.end.max(key.span.end));
                        entries.push(AttributeEntry {
                            key,
                            value,
                            span: entry_span,
                        });
                        self.skip_newlines();
                        if self.match_kind(TokenKind::RightParen) {
                            break;
                        }
                        self.expect(
                            TokenKind::Comma,
                            "WREN-P046",
                            "expected `,` or `)` in attribute group",
                        )?;
                        self.skip_newlines();
                        if self.match_kind(TokenKind::RightParen) {
                            break;
                        }
                    }
                }
                let end = self.previous().span.end;
                attributes.push(Attribute {
                    runtime,
                    group: Some(group),
                    entries,
                    span: Span::new(start, end),
                });
            } else {
                let key = key_or_group;
                let value = if self.match_kind(TokenKind::Equal) {
                    self.parse_attribute_value()?
                } else {
                    AttributeValue::Null
                };
                let end = self.previous().span.end.max(key.span.end);
                attributes.push(Attribute {
                    runtime,
                    group: None,
                    entries: vec![AttributeEntry {
                        key: key.clone(),
                        value,
                        span: Span::new(key.span.start, end),
                    }],
                    span: Span::new(start, end),
                });
            }

            self.expect(
                TokenKind::Newline,
                "WREN-P072",
                "expected newline after attribute",
            )?;
            self.skip_newlines();
        }
        Ok(attributes)
    }

    fn parse_attribute_value(&mut self) -> PResult<AttributeValue> {
        let token = self.peek();
        match token.kind {
            TokenKind::Identifier => {
                self.advance();
                Ok(AttributeValue::Name(self.slice(token.span).to_owned()))
            }
            TokenKind::True => {
                self.advance();
                Ok(AttributeValue::Bool(true))
            }
            TokenKind::False => {
                self.advance();
                Ok(AttributeValue::Bool(false))
            }
            TokenKind::Number => {
                self.advance();
                self.parse_number_token(token).map(AttributeValue::Number)
            }
            TokenKind::String | TokenKind::RawString => {
                self.advance();
                let (bytes, interpolated) = self.decode_string_for_import(token)?;
                if interpolated {
                    self.diagnostics.push(
                        Diagnostic::error(
                            "WREN-P047",
                            "attribute strings cannot use interpolation",
                            token.span,
                        )
                        .with_primary_label(token.span, "use a literal attribute value"),
                    );
                }
                Ok(AttributeValue::String(bytes))
            }
            _ => self.error_at_current(
                "WREN-P048",
                "invalid attribute value",
                "expected a name, string, boolean, or number literal",
            ),
        }
    }

    fn parse_expression(&mut self) -> PResult<Expr> {
        if self.recursion_depth >= MAX_PARSE_DEPTH {
            return self.error_at_current(
                "WREN-P049",
                "expression nesting is too deep",
                "simplify this expression",
            );
        }
        self.recursion_depth += 1;
        let result = self.parse_precedence(1);
        self.recursion_depth -= 1;
        result
    }

    fn parse_precedence(&mut self, min_bp: u8) -> PResult<Expr> {
        let mut left = self.parse_prefix()?;

        loop {
            // Wren permits exactly one line break immediately before a fluent `.`.
            if allows_line_before_dot(&left)
                && self.at(TokenKind::Newline)
                && self.tokens.get(self.current + 1).map(|token| token.kind)
                    == Some(TokenKind::Dot)
            {
                self.advance();
            }

            match self.peek().kind {
                TokenKind::Dot if 16 >= min_bp => {
                    self.advance();
                    self.skip_newlines();
                    left = self.parse_dot_call(left)?;
                    continue;
                }
                TokenKind::LeftBracket if 16 >= min_bp => {
                    self.advance();
                    left = self.parse_subscript(left)?;
                    continue;
                }
                TokenKind::LeftParen if 16 >= min_bp && matches!(&left.kind, ExprKind::Name(_)) => {
                    self.advance();
                    left = self.parse_implicit_call(left)?;
                    continue;
                }
                TokenKind::LeftBrace if 16 >= min_bp && self.can_attach_block_argument(&left) => {
                    self.advance();
                    left = self.attach_block_argument(left)?;
                    continue;
                }
                _ => {}
            }

            if self.at(TokenKind::Equal) {
                let lbp = 1;
                if lbp < min_bp {
                    break;
                }
                self.advance();
                let op_span = self.previous().span;
                self.skip_newlines();
                let right = self.parse_precedence(lbp)?;
                if !is_assignment_target(&left) {
                    self.diagnostics.push(
                        Diagnostic::error(
                            "WREN-P050",
                            "invalid assignment target",
                            left.span,
                        )
                        .with_primary_label(left.span, "this expression cannot be assigned to")
                        .with_secondary_label(op_span, "assignment occurs here"),
                    );
                }
                if let ExprKind::Subscript { args, .. } = &left.kind {
                    if args.len() >= MAX_PARAMETERS {
                        self.diagnostics.push(
                            Diagnostic::error(
                                "WREN-P073",
                                "subscript setter has too many arguments",
                                left.span,
                            )
                            .with_primary_label(
                                left.span,
                                format!(
                                    "{} index arguments plus the assigned value exceed the VM limit of {}",
                                    args.len(),
                                    MAX_PARAMETERS
                                ),
                            ),
                        );
                    }
                }
                let span = left.span.join(right.span);
                left = Expr::new(
                    ExprKind::Assign {
                        target: Box::new(left),
                        value: Box::new(right),
                    },
                    span,
                );
                continue;
            }

            if self.at(TokenKind::Question) {
                let lbp = 2;
                if lbp < min_bp {
                    break;
                }
                self.advance();
                self.skip_newlines();
                let then_expr = self.parse_precedence(lbp)?;
                self.skip_newlines();
                self.expect(
                    TokenKind::Colon,
                    "WREN-P051",
                    "expected `:` in conditional expression",
                )?;
                self.skip_newlines();
                let else_expr = self.parse_precedence(1)?;
                let span = left.span.join(else_expr.span);
                left = Expr::new(
                    ExprKind::Conditional {
                        condition: Box::new(left),
                        then_expr: Box::new(then_expr),
                        else_expr: Box::new(else_expr),
                    },
                    span,
                );
                continue;
            }

            let Some((lbp, rbp, infix)) = infix_binding(self.peek().kind) else {
                break;
            };
            if lbp < min_bp {
                break;
            }

            self.advance();
            self.skip_newlines();
            let right = self.parse_precedence(rbp)?;
            let span = left.span.join(right.span);
            left = match infix {
                InfixKind::Binary(operator) => Expr::new(
                    ExprKind::Binary {
                        left: Box::new(left),
                        operator,
                        right: Box::new(right),
                    },
                    span,
                ),
                InfixKind::Logical(operator) => Expr::new(
                    ExprKind::Logical {
                        left: Box::new(left),
                        operator,
                        right: Box::new(right),
                    },
                    span,
                ),
            };
        }

        Ok(left)
    }

    fn parse_prefix(&mut self) -> PResult<Expr> {
        let token = self.advance();
        match token.kind {
            TokenKind::Null => Ok(Expr::new(ExprKind::Null, token.span)),
            TokenKind::True => Ok(Expr::new(ExprKind::Bool(true), token.span)),
            TokenKind::False => Ok(Expr::new(ExprKind::Bool(false), token.span)),
            TokenKind::Number => {
                let value = self.parse_number_token(token)?;
                Ok(Expr::new(ExprKind::Number(value), token.span))
            }
            TokenKind::String => self.parse_normal_string_expr(token),
            TokenKind::RawString => {
                let bytes = decode_raw_string(self.slice(token.span));
                Ok(Expr::new(ExprKind::String(bytes), token.span))
            }
            TokenKind::Identifier => {
                let ident = Spanned::new(self.slice(token.span).to_owned(), token.span);
                Ok(Expr::new(ExprKind::Name(ident), token.span))
            }
            TokenKind::Field => {
                let ident = Spanned::new(self.slice(token.span).to_owned(), token.span);
                Ok(Expr::new(ExprKind::Field(ident), token.span))
            }
            TokenKind::StaticField => {
                let ident = Spanned::new(self.slice(token.span).to_owned(), token.span);
                Ok(Expr::new(ExprKind::StaticField(ident), token.span))
            }
            TokenKind::This => Ok(Expr::new(ExprKind::This, token.span)),
            TokenKind::Super => self.parse_super(token.span.start),
            TokenKind::Minus | TokenKind::Bang | TokenKind::Tilde => {
                let operator = match token.kind {
                    TokenKind::Minus => UnaryOp::Negate,
                    TokenKind::Bang => UnaryOp::Not,
                    TokenKind::Tilde => UnaryOp::Complement,
                    _ => unreachable!(),
                };
                self.skip_newlines();
                let operand = self.parse_precedence(15)?;
                let span = token.span.join(operand.span);
                Ok(Expr::new(
                    ExprKind::Unary {
                        operator,
                        operand: Box::new(operand),
                    },
                    span,
                ))
            }
            TokenKind::LeftParen => {
                let expression = self.parse_expression()?;
                let close = self.expect(
                    TokenKind::RightParen,
                    "WREN-P052",
                    "expected `)` after expression",
                )?;
                Ok(Expr::new(
                    ExprKind::Grouped(Box::new(expression)),
                    Span::new(token.span.start, close.span.end),
                ))
            }
            TokenKind::LeftBracket => self.parse_list_literal(token.span.start),
            TokenKind::LeftBrace => self.parse_map_literal(token.span.start),
            _ => {
                self.current = self.current.saturating_sub(1);
                self.error_at_current(
                    "WREN-P053",
                    "expected expression",
                    format!("found {}", token.kind.description()),
                )
            }
        }
    }

    fn parse_dot_call(&mut self, receiver: Expr) -> PResult<Expr> {
        let name = self.parse_identifier("WREN-P054", "expected method name after `.`")?;
        let (syntax, mut args, end) = if self.match_kind(TokenKind::LeftParen) {
            let args = self.parse_argument_list(TokenKind::RightParen)?;
            (InvokeSyntax::Method, args, self.previous().span.end)
        } else {
            (InvokeSyntax::Getter, Vec::new(), name.span.end)
        };

        let span = Span::new(receiver.span.start, end);
        Ok(Expr::new(
            ExprKind::Invoke(InvokeExpr {
                receiver: Some(Box::new(receiver)),
                name,
                args: std::mem::take(&mut args),
                syntax,
            }),
            span,
        ))
    }

    fn parse_implicit_call(&mut self, name_expr: Expr) -> PResult<Expr> {
        let start = name_expr.span.start;
        let ExprKind::Name(name) = name_expr.kind else {
            unreachable!();
        };
        let args = self.parse_argument_list(TokenKind::RightParen)?;
        let end = self.previous().span.end;
        Ok(Expr::new(
            ExprKind::Invoke(InvokeExpr {
                receiver: None,
                name,
                args,
                syntax: InvokeSyntax::Method,
            }),
            Span::new(start, end),
        ))
    }

    fn parse_subscript(&mut self, receiver: Expr) -> PResult<Expr> {
        let args = self.parse_argument_list(TokenKind::RightBracket)?;
        if args.is_empty() {
            self.diagnostics.push(
                Diagnostic::error(
                    "WREN-P055",
                    "a subscript expression needs at least one argument",
                    self.previous().span,
                )
                .with_primary_label(self.previous().span, "empty subscript"),
            );
        }
        let end = self.previous().span.end;
        let start = receiver.span.start;
        Ok(Expr::new(
            ExprKind::Subscript {
                receiver: Box::new(receiver),
                args,
            },
            Span::new(start, end),
        ))
    }

    fn parse_argument_list(&mut self, close: TokenKind) -> PResult<Vec<Expr>> {
        let mut args = Vec::new();
        self.skip_newlines();
        if self.match_kind(close) {
            return Ok(args);
        }

        loop {
            self.skip_newlines();
            args.push(self.parse_expression()?);

            if self.match_kind(TokenKind::Comma) {
                continue;
            }

            // A newline is allowed before the closing delimiter, but not before
            // a comma. This matches finishArgumentList() in Wren's compiler.
            self.skip_newlines();
            self.expect(
                close,
                "WREN-P056",
                format!("expected `,` or {} after argument", close.description()),
            )?;
            break;
        }

        if args.len() > MAX_PARAMETERS {
            self.diagnostics.push(
                Diagnostic::error(
                    "WREN-P057",
                    format!("call has {} arguments, maximum is {}", args.len(), MAX_PARAMETERS),
                    Span::new(
                        args.first().map_or(self.previous().span.start, |arg| arg.span.start),
                        self.previous().span.end,
                    ),
                )
                .with_note("The bytecode ABI has ARG0..ARG15."),
            );
        }
        Ok(args)
    }

    fn can_attach_block_argument(&self, expression: &Expr) -> bool {
        matches!(
            &expression.kind,
            ExprKind::Name(_) | ExprKind::Invoke(_) | ExprKind::Super(_)
        )
    }

    fn attach_block_argument(&mut self, mut expression: Expr) -> PResult<Expr> {
        let body = self.parse_function_body_after_open(self.previous().span.start, true)?;
        let end = body.span.end;
        let closure = Expr::new(ExprKind::Closure(body.clone()), body.span);

        match &mut expression.kind {
            ExprKind::Name(name) => {
                let name = name.clone();
                expression.kind = ExprKind::Invoke(InvokeExpr {
                    receiver: None,
                    name,
                    args: vec![closure],
                    syntax: InvokeSyntax::Method,
                });
            }
            ExprKind::Invoke(invoke) => {
                invoke.syntax = InvokeSyntax::Method;
                invoke.args.push(closure);
                if invoke.args.len() > MAX_PARAMETERS {
                    self.diagnostics.push(
                        Diagnostic::error(
                            "WREN-P058",
                            "block argument exceeds the maximum method arity",
                            body.span,
                        )
                        .with_note("A block argument is the final explicit method argument."),
                    );
                }
            }
            ExprKind::Super(super_expr) => {
                super_expr.syntax = InvokeSyntax::Method;
                super_expr.args.push(closure);
                if super_expr.args.len() > MAX_PARAMETERS {
                    self.diagnostics.push(
                        Diagnostic::error(
                            "WREN-P058",
                            "block argument exceeds the maximum method arity",
                            body.span,
                        )
                        .with_note("A block argument is the final explicit method argument."),
                    );
                }
            }
            _ => unreachable!(),
        }
        expression.span.end = end;
        Ok(expression)
    }

    fn parse_super(&mut self, start: usize) -> PResult<Expr> {
        if self.match_kind(TokenKind::Dot) {
            let method = self.parse_identifier(
                "WREN-P059",
                "expected superclass method name after `super.`",
            )?;
            let (syntax, args, end) = if self.match_kind(TokenKind::LeftParen) {
                let args = self.parse_argument_list(TokenKind::RightParen)?;
                (InvokeSyntax::Method, args, self.previous().span.end)
            } else {
                (InvokeSyntax::Getter, Vec::new(), method.span.end)
            };
            return Ok(Expr::new(
                ExprKind::Super(SuperExpr {
                    method: Some(method),
                    args,
                    syntax,
                }),
                Span::new(start, end),
            ));
        }

        if self.match_kind(TokenKind::LeftParen) {
            let args = self.parse_argument_list(TokenKind::RightParen)?;
            let end = self.previous().span.end;
            return Ok(Expr::new(
                ExprKind::Super(SuperExpr {
                    method: None,
                    args,
                    syntax: InvokeSyntax::Method,
                }),
                Span::new(start, end),
            ));
        }

        // With no explicit name or argument list, Wren dispatches to a getter
        // with the same name as the enclosing method. The lowerer knows that
        // enclosing method and resolves the actual symbol.
        Ok(Expr::new(
            ExprKind::Super(SuperExpr {
                method: None,
                args: Vec::new(),
                syntax: InvokeSyntax::Getter,
            }),
            Span::new(start, self.previous().span.end),
        ))
    }

    fn parse_list_literal(&mut self, start: usize) -> PResult<Expr> {
        let mut values = Vec::new();

        loop {
            self.skip_newlines();
            if self.match_kind(TokenKind::RightBracket) {
                break;
            }

            values.push(self.parse_expression()?);
            if self.match_kind(TokenKind::Comma) {
                continue;
            }

            // Newlines are allowed before `]`, but not before a comma.
            self.skip_newlines();
            self.expect(
                TokenKind::RightBracket,
                "WREN-P061",
                "expected `,` or `]` in list literal",
            )?;
            break;
        }

        Ok(Expr::new(
            ExprKind::List(values),
            Span::new(start, self.previous().span.end),
        ))
    }

    fn parse_map_literal(&mut self, start: usize) -> PResult<Expr> {
        let mut entries = Vec::new();

        loop {
            self.skip_newlines();
            if self.match_kind(TokenKind::RightBrace) {
                break;
            }

            // Wren parses map keys at unary precedence. Parenthesize a key if
            // it needs lower-precedence binary operators.
            let key = self.parse_precedence(15)?;
            self.expect(
                TokenKind::Colon,
                "WREN-P062",
                "expected `:` between map key and value",
            )?;
            self.skip_newlines();
            let value = self.parse_expression()?;
            let span = key.span.join(value.span);
            entries.push(MapEntry { key, value, span });

            if self.match_kind(TokenKind::Comma) {
                continue;
            }

            // Newlines are allowed before `}`, but not before a comma.
            self.skip_newlines();
            self.expect(
                TokenKind::RightBrace,
                "WREN-P063",
                "expected `,` or `}` in map literal",
            )?;
            break;
        }

        Ok(Expr::new(
            ExprKind::Map(entries),
            Span::new(start, self.previous().span.end),
        ))
    }

    fn parse_normal_string_expr(&mut self, token: Token) -> PResult<Expr> {
        let lexeme = self.slice(token.span);
        debug_assert!(lexeme.starts_with('"') && lexeme.ends_with('"'));
        let content_start = token.span.start + 1;
        let content_end = token.span.end.saturating_sub(1);
        let content = self.source.as_bytes()[content_start..content_end].to_vec();
        let bytes = content.as_slice();

        let mut parts = Vec::new();
        let mut segment_start = 0usize;
        let mut index = 0usize;
        let mut saw_interpolation = false;

        while index < bytes.len() {
            if bytes[index] == b'\\' {
                index += 1;
                if index < bytes.len() {
                    index += 1;
                }
                continue;
            }

            if bytes[index] == b'%' && index + 1 < bytes.len() && bytes[index + 1] == b'(' {
                saw_interpolation = true;
                let literal = self.decode_escaped_segment(
                    &bytes[segment_start..index],
                    content_start + segment_start,
                );
                if !literal.is_empty() {
                    parts.push(StringPart::Bytes(literal));
                }

                let interpolation_body = index + 2;
                let consumed = match scan_interpolation_tail(&bytes[interpolation_body..], 1) {
                    Ok(consumed) => consumed,
                    Err(_) => {
                        self.diagnostics.push(
                            Diagnostic::error(
                                "WREN-P064",
                                "unterminated string interpolation",
                                token.span,
                            )
                            .with_primary_label(token.span, "interpolation starts inside this string"),
                        );
                        return Ok(Expr::new(ExprKind::InterpolatedString(parts), token.span));
                    }
                };

                let body_end = interpolation_body + consumed - 1;
                let absolute_start = content_start + interpolation_body;
                let absolute_end = content_start + body_end;
                let expression = self.parse_interpolation_expression(absolute_start, absolute_end)?;
                parts.push(StringPart::Expression(expression));

                index = interpolation_body + consumed;
                segment_start = index;
                continue;
            }

            index += 1;
        }

        let tail = self.decode_escaped_segment(
            &bytes[segment_start..],
            content_start + segment_start,
        );
        if !tail.is_empty() || parts.is_empty() {
            parts.push(StringPart::Bytes(tail));
        }

        if !saw_interpolation {
            let bytes = match parts.pop() {
                Some(StringPart::Bytes(bytes)) => bytes,
                _ => Vec::new(),
            };
            Ok(Expr::new(ExprKind::String(bytes), token.span))
        } else {
            Ok(Expr::new(ExprKind::InterpolatedString(parts), token.span))
        }
    }

    fn parse_interpolation_expression(
        &mut self,
        absolute_start: usize,
        absolute_end: usize,
    ) -> PResult<Expr> {
        if absolute_start > absolute_end || absolute_end > self.source.len() {
            return self.error_at(
                "WREN-P065",
                "invalid interpolation source range",
                Span::new(absolute_start, absolute_end.min(self.source.len())),
                "internal interpolation boundary error",
            );
        }

        let substring = &self.source[absolute_start..absolute_end];
        let lexed = lex_with_offset(substring, absolute_start);
        self.diagnostics.extend(lexed.diagnostics);

        let mut nested = Parser::new(self.source_name, self.source, lexed.tokens);
        nested.recursion_depth = self.recursion_depth + 1;
        let result = nested.parse_expression_only();
        let nested_diagnostics = std::mem::take(&mut nested.diagnostics);
        drop(nested);
        self.diagnostics.extend(nested_diagnostics);
        result
    }

    fn parse_expression_only(&mut self) -> PResult<Expr> {
        self.skip_newlines();
        if self.at(TokenKind::Eof) {
            return self.error_at_current(
                "WREN-P066",
                "empty string interpolation",
                "expected an expression inside `%(` and `)`",
            );
        }
        let expression = self.parse_expression()?;
        self.skip_newlines();
        if !self.at(TokenKind::Eof) {
            self.diagnostics.push(
                Diagnostic::error(
                    "WREN-P067",
                    "unexpected tokens after interpolation expression",
                    self.peek().span,
                )
                .with_primary_label(self.peek().span, "expression should end before this token"),
            );
        }
        Ok(expression)
    }

    fn decode_string_for_import(&mut self, token: Token) -> PResult<(Vec<u8>, bool)> {
        if token.kind == TokenKind::RawString {
            return Ok((decode_raw_string(self.slice(token.span)), false));
        }

        let content_start = token.span.start + 1;
        let content_end = token.span.end.saturating_sub(1);
        let content = self.source.as_bytes()[content_start..content_end].to_vec();
        let bytes = content.as_slice();
        let mut output = Vec::new();
        let mut segment_start = 0usize;
        let mut index = 0usize;
        let mut interpolated = false;

        while index < bytes.len() {
            if bytes[index] == b'\\' {
                index += 1;
                if index < bytes.len() {
                    index += 1;
                }
                continue;
            }
            if bytes[index] == b'%' && index + 1 < bytes.len() && bytes[index + 1] == b'(' {
                interpolated = true;
                output.extend(self.decode_escaped_segment(
                    &bytes[segment_start..index],
                    content_start + segment_start,
                ));
                let body_start = index + 2;
                let consumed = scan_interpolation_tail(&bytes[body_start..], 1).map_err(|_| ())?;
                index = body_start + consumed;
                segment_start = index;
                continue;
            }
            index += 1;
        }

        output.extend(self.decode_escaped_segment(
            &bytes[segment_start..],
            content_start + segment_start,
        ));
        Ok((output, interpolated))
    }

    fn decode_escaped_segment(&mut self, bytes: &[u8], absolute_start: usize) -> Vec<u8> {
        let mut output = Vec::with_capacity(bytes.len());
        let mut index = 0usize;

        while index < bytes.len() {
            // The reference Wren lexer ignores carriage returns in strings.
            // In CRLF input the following LF is then preserved normally.
            if bytes[index] == b'\r' {
                index += 1;
                continue;
            }

            if bytes[index] != b'\\' {
                output.push(bytes[index]);
                index += 1;
                continue;
            }

            let escape_start = index;
            index += 1;
            if index >= bytes.len() {
                self.invalid_escape(
                    Span::new(absolute_start + escape_start, absolute_start + index),
                    "backslash at end of string",
                );
                break;
            }

            match bytes[index] {
                b'0' => {
                    output.push(0);
                    index += 1;
                }
                b'"' => {
                    output.push(b'"');
                    index += 1;
                }
                b'\\' => {
                    output.push(b'\\');
                    index += 1;
                }
                b'%' => {
                    output.push(b'%');
                    index += 1;
                }
                b'a' => {
                    output.push(0x07);
                    index += 1;
                }
                b'b' => {
                    output.push(0x08);
                    index += 1;
                }
                b'e' => {
                    output.push(0x1b);
                    index += 1;
                }
                b'f' => {
                    output.push(0x0c);
                    index += 1;
                }
                b'n' => {
                    output.push(b'\n');
                    index += 1;
                }
                b'r' => {
                    output.push(b'\r');
                    index += 1;
                }
                b't' => {
                    output.push(b'\t');
                    index += 1;
                }
                b'v' => {
                    output.push(0x0b);
                    index += 1;
                }
                b'x' => {
                    index += 1;
                    match parse_fixed_hex(bytes, index, 2) {
                        Some(value) => {
                            output.push(value as u8);
                            index += 2;
                        }
                        None => {
                            let end = (index + 2).min(bytes.len());
                            self.invalid_escape(
                                Span::new(
                                    absolute_start + escape_start,
                                    absolute_start + end,
                                ),
                                "`\\x` escape requires exactly two hexadecimal digits",
                            );
                            index = end;
                        }
                    }
                }
                b'u' | b'U' => {
                    let digits = if bytes[index] == b'u' { 4 } else { 8 };
                    index += 1;
                    match parse_fixed_hex(bytes, index, digits) {
                        Some(value) => {
                            if let Some(character) = char::from_u32(value) {
                                let mut buffer = [0u8; 4];
                                output.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
                            } else {
                                self.invalid_escape(
                                    Span::new(
                                        absolute_start + escape_start,
                                        absolute_start + index + digits,
                                    ),
                                    "escape is not a valid Unicode scalar value",
                                );
                            }
                            index += digits;
                        }
                        None => {
                            let end = (index + digits).min(bytes.len());
                            self.invalid_escape(
                                Span::new(
                                    absolute_start + escape_start,
                                    absolute_start + end,
                                ),
                                format!("Unicode escape requires exactly {digits} hexadecimal digits"),
                            );
                            index = end;
                        }
                    }
                }
                other => {
                    let end = index + 1;
                    self.invalid_escape(
                        Span::new(
                            absolute_start + escape_start,
                            absolute_start + end,
                        ),
                        format!("unknown escape `\\{}`", other as char),
                    );
                    output.push(other);
                    index = end;
                }
            }
        }

        output
    }

    fn invalid_escape(&mut self, span: Span, message: impl Into<String>) {
        self.diagnostics.push(
            Diagnostic::error("WREN-P068", "invalid string escape", span)
                .with_primary_label(span, message),
        );
    }

    fn parse_number_token(&mut self, token: Token) -> PResult<f64> {
        let text = self.slice(token.span);
        let value = if text.starts_with("0x") {
            u64::from_str_radix(&text[2..], 16).ok().map(|value| value as f64)
        } else {
            text.parse::<f64>().ok()
        };

        match value {
            Some(value) if value.is_finite() => Ok(value),
            _ => self.error_at(
                "WREN-P069",
                "invalid number literal",
                token.span,
                "this number cannot be represented as a Wren Num",
            ),
        }
    }

    fn parse_identifier(&mut self, code: &'static str, message: impl Into<String>) -> PResult<Ident> {
        if !self.at(TokenKind::Identifier) {
            return self.error_at_current(code, message, "expected an identifier here");
        }
        let token = self.advance();
        Ok(Spanned::new(self.slice(token.span).to_owned(), token.span))
    }

    fn peek_operator_name(&self) -> Option<OperatorName> {
        Some(match self.peek().kind {
            TokenKind::Bang => OperatorName::Bang,
            TokenKind::Tilde => OperatorName::Tilde,
            TokenKind::Minus => OperatorName::Minus,
            TokenKind::Plus => OperatorName::Plus,
            TokenKind::Star => OperatorName::Star,
            TokenKind::Slash => OperatorName::Slash,
            TokenKind::Percent => OperatorName::Percent,
            TokenKind::DotDot => OperatorName::DotDot,
            TokenKind::DotDotDot => OperatorName::DotDotDot,
            TokenKind::ShiftLeft => OperatorName::ShiftLeft,
            TokenKind::ShiftRight => OperatorName::ShiftRight,
            TokenKind::Ampersand => OperatorName::Ampersand,
            TokenKind::Caret => OperatorName::Caret,
            TokenKind::Pipe => OperatorName::Pipe,
            TokenKind::Less => OperatorName::Less,
            TokenKind::LessEqual => OperatorName::LessEqual,
            TokenKind::Greater => OperatorName::Greater,
            TokenKind::GreaterEqual => OperatorName::GreaterEqual,
            TokenKind::EqualEqual => OperatorName::EqualEqual,
            TokenKind::BangEqual => OperatorName::BangEqual,
            _ => return None,
        })
    }

    #[inline]
    fn peek(&self) -> Token {
        self.tokens
            .get(self.current)
            .copied()
            .unwrap_or_else(|| *self.tokens.last().expect("token stream always has EOF"))
    }

    #[inline]
    fn previous(&self) -> Token {
        self.tokens[self.current.saturating_sub(1)]
    }

    #[inline]
    fn at(&self, kind: TokenKind) -> bool {
        self.peek().kind == kind
    }

    #[inline]
    fn advance(&mut self) -> Token {
        let token = self.peek();
        if token.kind != TokenKind::Eof {
            self.current += 1;
        }
        token
    }

    #[inline]
    fn match_kind(&mut self, kind: TokenKind) -> bool {
        if self.at(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(
        &mut self,
        kind: TokenKind,
        code: &'static str,
        message: impl Into<String>,
    ) -> PResult<Token> {
        if self.at(kind) {
            Ok(self.advance())
        } else {
            let found = self.peek();
            let message = message.into();
            self.diagnostics.push(
                Diagnostic::error(code, message, found.span).with_primary_label(
                    found.span,
                    format!("found {}, expected {}", found.kind.description(), kind.description()),
                ),
            );
            Err(())
        }
    }

    fn error_at_current<T>(
        &mut self,
        code: &'static str,
        message: impl Into<String>,
        label: impl Into<String>,
    ) -> PResult<T> {
        let span = self.peek().span;
        self.error_at(code, message, span, label)
    }

    fn error_at<T>(
        &mut self,
        code: &'static str,
        message: impl Into<String>,
        span: Span,
        label: impl Into<String>,
    ) -> PResult<T> {
        self.diagnostics.push(
            Diagnostic::error(code, message, span).with_primary_label(span, label),
        );
        Err(())
    }

    #[inline]
    fn slice(&self, span: Span) -> &str {
        &self.source[span.start..span.end]
    }

    fn skip_newlines(&mut self) {
        while self.match_kind(TokenKind::Newline) {}
    }

    fn consume_required_line(&mut self, code: &'static str, message: &'static str) {
        if self.at(TokenKind::Newline) {
            self.skip_newlines();
            return;
        }

        let span = self.peek().span;
        self.diagnostics.push(
            Diagnostic::error(code, message, span)
                .with_primary_label(span, "a newline is required before this token"),
        );

        if !self.at(TokenKind::RightBrace) && !self.at(TokenKind::Eof) {
            self.synchronize_statement();
            self.skip_newlines();
        }
    }

    fn consume_statement_boundary(&mut self) {
        if self.at(TokenKind::Newline) {
            self.skip_newlines();
            return;
        }
        if self.at(TokenKind::RightBrace) || self.at(TokenKind::Eof) {
            return;
        }

        let span = self.peek().span;
        self.diagnostics.push(
            Diagnostic::error(
                "WREN-P070",
                "expected a newline between statements",
                span,
            )
            .with_primary_label(span, "this token starts before the previous statement ended"),
        );
        self.synchronize_statement();
    }

    fn synchronize_statement(&mut self) {
        while !self.at(TokenKind::Eof)
            && !self.at(TokenKind::RightBrace)
            && !self.at(TokenKind::Newline)
        {
            self.advance();
        }
        if self.at(TokenKind::Newline) {
            self.skip_newlines();
        }
    }

    fn synchronize_class_member(&mut self) {
        while !self.at(TokenKind::Eof)
            && !self.at(TokenKind::RightBrace)
            && !self.at(TokenKind::Newline)
        {
            self.advance();
        }
        self.skip_newlines();
    }
}

#[derive(Clone, Copy, Debug)]
enum InfixKind {
    Binary(BinaryOp),
    Logical(LogicalOp),
}

fn infix_binding(kind: TokenKind) -> Option<(u8, u8, InfixKind)> {
    use BinaryOp as B;
    use InfixKind::{Binary, Logical};
    use LogicalOp as L;
    use TokenKind as T;

    let (bp, operation) = match kind {
        T::OrOr => (3, Logical(L::Or)),
        T::AndAnd => (4, Logical(L::And)),
        T::EqualEqual => (5, Binary(B::Equal)),
        T::BangEqual => (5, Binary(B::NotEqual)),
        T::Is => (6, Binary(B::Is)),
        T::Less => (7, Binary(B::Less)),
        T::LessEqual => (7, Binary(B::LessEqual)),
        T::Greater => (7, Binary(B::Greater)),
        T::GreaterEqual => (7, Binary(B::GreaterEqual)),
        T::Pipe => (8, Binary(B::BitOr)),
        T::Caret => (9, Binary(B::BitXor)),
        T::Ampersand => (10, Binary(B::BitAnd)),
        T::ShiftLeft => (11, Binary(B::ShiftLeft)),
        T::ShiftRight => (11, Binary(B::ShiftRight)),
        T::DotDot => (12, Binary(B::InclusiveRange)),
        T::DotDotDot => (12, Binary(B::ExclusiveRange)),
        T::Plus => (13, Binary(B::Add)),
        T::Minus => (13, Binary(B::Subtract)),
        T::Star => (14, Binary(B::Multiply)),
        T::Slash => (14, Binary(B::Divide)),
        T::Percent => (14, Binary(B::Modulo)),
        _ => return None,
    };

    // All Wren binary operators associate left.
    Some((bp, bp + 1, operation))
}


fn allows_line_before_dot(expression: &Expr) -> bool {
    match &expression.kind {
        ExprKind::Name(_)
        | ExprKind::Field(_)
        | ExprKind::StaticField(_)
        | ExprKind::Invoke(_)
        | ExprKind::Subscript { .. } => true,
        ExprKind::Super(super_expr) => super_expr.method.is_some(),
        _ => false,
    }
}

fn is_assignment_target(expression: &Expr) -> bool {
    match &expression.kind {
        ExprKind::Name(_) | ExprKind::Field(_) | ExprKind::StaticField(_) | ExprKind::Subscript { .. } => true,
        ExprKind::Invoke(invoke) => invoke.syntax == InvokeSyntax::Getter,
        ExprKind::Super(super_expr) => super_expr.syntax == InvokeSyntax::Getter,
        _ => false,
    }
}

fn parse_fixed_hex(bytes: &[u8], start: usize, digits: usize) -> Option<u32> {
    if start + digits > bytes.len() {
        return None;
    }
    let mut value = 0u32;
    for &byte in &bytes[start..start + digits] {
        let digit = match byte {
            b'0'..=b'9' => (byte - b'0') as u32,
            b'a'..=b'f' => (byte - b'a' + 10) as u32,
            b'A'..=b'F' => (byte - b'A' + 10) as u32,
            _ => return None,
        };
        value = (value << 4) | digit;
    }
    Some(value)
}

fn decode_raw_string(lexeme: &str) -> Vec<u8> {
    if lexeme.len() < 6 {
        return Vec::new();
    }
    let raw = &lexeme.as_bytes()[3..lexeme.len() - 3];
    let normalized = normalize_newlines(raw);
    trim_raw_string_delimiter_lines(normalized)
}

fn normalize_newlines(bytes: &[u8]) -> Vec<u8> {
    // Wren ignores CR bytes in raw strings instead of normalizing them.
    // This preserves LF in CRLF input and removes lone CR exactly like the
    // reference lexer.
    bytes.iter().copied().filter(|&byte| byte != b'\r').collect()
}

fn trim_raw_string_delimiter_lines(mut bytes: Vec<u8>) -> Vec<u8> {
    if let Some(first_newline) = bytes.iter().position(|&byte| byte == b'\n') {
        if bytes[..first_newline]
            .iter()
            .all(|byte| matches!(byte, b' ' | b'\t'))
        {
            bytes.drain(..=first_newline);
        }
    }

    if let Some(last_newline) = bytes.iter().rposition(|&byte| byte == b'\n') {
        if bytes[last_newline + 1..]
            .iter()
            .all(|byte| matches!(byte, b' ' | b'\t'))
        {
            bytes.truncate(last_newline);
        }
    }

    bytes
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(source: &str) -> ModuleAst {
        let output = parse("test.wren", source);
        if output.has_errors() {
            for diagnostic in &output.diagnostics {
                let _ = diagnostic.eprint("test.wren", source);
            }
            panic!("frontend emitted errors");
        }
        output.module
    }

    #[test]
    fn lexer_treats_nested_block_comments_as_whitespace() {
        let source = "var a = 1 /* outer\n /* inner */\n end */ var b = 2";
        let output = lex(source);
        assert!(output.diagnostics.is_empty());
        let newlines = output
            .tokens
            .iter()
            .filter(|token| token.kind == TokenKind::Newline)
            .count();
        assert_eq!(newlines, 0);
    }

    #[test]
    fn lexer_distinguishes_fields_static_fields_and_names() {
        let output = lex("_field __staticField ordinary");
        assert!(output.diagnostics.is_empty());
        let kinds = output
            .tokens
            .iter()
            .map(|token| token.kind)
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            vec![
                TokenKind::Field,
                TokenKind::StaticField,
                TokenKind::Identifier,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn lexer_accepts_utf8_bom_at_source_start() {
        let output = lex("\u{feff}var value = 1\n");
        assert!(output.diagnostics.is_empty());
        assert_eq!(output.tokens[0].kind, TokenKind::Var);
        assert_eq!(output.tokens[0].span.start, 3);
    }


    #[test]
    fn lexer_accepts_unix_shebang_only_at_source_start() {
        let output = lex("#!/usr/bin/env wren\nvar value = 1\n");
        assert!(output.diagnostics.is_empty());
        assert_eq!(output.tokens[0].kind, TokenKind::Newline);
        assert_eq!(output.tokens[1].kind, TokenKind::Var);

        let output = lex("var value = 1\n#!/usr/bin/env wren\n");
        assert!(!output.diagnostics.is_empty());
    }

    #[test]
    fn lexer_enforces_official_interpolation_depth_limit() {
        fn nested(level: usize) -> String {
            if level == 0 {
                return "\"leaf\"".to_owned();
            }
            format!("\"%({})\"", nested(level - 1))
        }

        let accepted = lex(&nested(MAX_INTERPOLATION_DEPTH));
        assert!(accepted.diagnostics.is_empty());

        let rejected = lex(&nested(MAX_INTERPOLATION_DEPTH + 1));
        assert!(rejected
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "WREN-L004"));
    }

    #[test]
    fn lexer_reports_incomplete_scientific_notation() {
        let output = lex("var value = 1e+\n");
        assert!(output
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "WREN-L007"));
    }

    #[test]
    fn parses_operator_precedence() {
        let module = parse_ok("var x = 1 + 2 * 3 == 7 || false\n");
        let StmtKind::Variable {
            initializer: Some(expr),
            ..
        } = &module.statements[0].kind
        else {
            panic!("expected variable");
        };
        assert!(matches!(
            expr.kind,
            ExprKind::Logical {
                operator: LogicalOp::Or,
                ..
            }
        ));
    }


    #[test]
    fn conditional_else_branch_keeps_assignment_precedence() {
        let module = parse_ok("var result = condition ? yes : target = value\n");
        let StmtKind::Variable {
            initializer: Some(expr),
            ..
        } = &module.statements[0].kind
        else {
            panic!("expected variable");
        };

        let ExprKind::Conditional { else_expr, .. } = &expr.kind else {
            panic!("expected conditional expression");
        };
        assert!(matches!(else_expr.kind, ExprKind::Assign { .. }));
    }

    #[test]
    fn conditional_then_branch_rejects_assignment_without_grouping() {
        let output = parse("test.wren", "var result = condition ? target = value : other\n");
        assert!(output.has_errors());
    }

    #[test]
    fn map_binary_keys_require_grouping_like_reference_wren() {
        let output = parse("test.wren", "var bad = {1 + 2: 3}\n");
        assert!(output.has_errors());

        let output = parse("test.wren", "var good = {(1 + 2): 3}\n");
        assert!(!output.has_errors());
    }

    #[test]
    fn grouping_does_not_ignore_newlines() {
        let output = parse("test.wren", "var value = (\n1)\n");
        assert!(output.has_errors());
    }

    #[test]
    fn fluent_dot_is_only_allowed_after_supported_receiver_forms() {
        let accepted = parse("test.wren", "var result = receiver\n.method()\n");
        assert!(!accepted.has_errors());

        let rejected_literal = parse("test.wren", "var result = 1\n.method()\n");
        assert!(rejected_literal.has_errors());

        let rejected_group = parse("test.wren", "var result = (receiver)\n.method()\n");
        assert!(rejected_group.has_errors());
    }

    #[test]
    fn block_statement_distinguishes_expression_and_statement_forms() {
        let expression_block = parse("test.wren", "if (true) { System.print(1) }\n");
        assert!(!expression_block.has_errors());

        let invalid_definition_block = parse("test.wren", "if (true) { var x = 1 }\n");
        assert!(invalid_definition_block.has_errors());

        let statement_block = parse("test.wren", "if (true) {\nvar x = 1\n}\n");
        assert!(!statement_block.has_errors());
    }

    #[test]
    fn import_allows_newline_after_keyword() {
        let module = parse_ok("import\n\"module\" for Item\n");
        assert!(matches!(module.statements[0].kind, StmtKind::Import(_)));
    }

    #[test]
    fn method_parameter_delimiter_newlines_match_wren() {
        let accepted = parse(
            "test.wren",
            "class Example {\nmethod(\nvalue) { value }\n}\n",
        );
        assert!(!accepted.has_errors());

        let rejected = parse(
            "test.wren",
            "class Example {\nmethod(value\n) { value }\n}\n",
        );
        assert!(rejected.has_errors());
    }

    #[test]
    fn parses_expression_and_statement_function_bodies_differently() {
        let source = r#"
class Example {
  expr { 40 + 2 }
  block {
    var x = 40
    return x + 2
  }
}
"#;
        let module = parse_ok(source);
        let StmtKind::Class(class) = &module.statements[0].kind else {
            panic!("expected class");
        };
        assert!(matches!(
            class.methods[0].body.as_ref().unwrap().kind,
            FunctionBodyKind::Expression(_)
        ));
        assert!(matches!(
            class.methods[1].body.as_ref().unwrap().kind,
            FunctionBodyKind::Statements(_)
        ));
    }

    #[test]
    fn parses_all_wren_method_signature_shapes() {
        let source = r#"
class Example {
  getter { 1 }
  method(a, b) { a + b }
  setter=(value) { value }
  - { 1 }
  +(other) { other }
  [index] { index }
  [x, y]=(value) { value }
  construct new(value) { _value = value }
  static make() { Example.new(1) }
  foreign static nativeCall(arg)
}
"#;
        let module = parse_ok(source);
        let StmtKind::Class(class) = &module.statements[0].kind else {
            panic!("expected class");
        };
        assert_eq!(class.methods.len(), 10);
        assert!(matches!(class.methods[0].signature, MethodSignatureAst::Getter { .. }));
        assert!(matches!(class.methods[1].signature, MethodSignatureAst::Method { .. }));
        assert!(matches!(class.methods[2].signature, MethodSignatureAst::Setter { .. }));
        assert!(matches!(class.methods[3].signature, MethodSignatureAst::PrefixOperator { .. }));
        assert!(matches!(class.methods[4].signature, MethodSignatureAst::InfixOperator { .. }));
        assert!(matches!(class.methods[5].signature, MethodSignatureAst::SubscriptGetter { .. }));
        assert!(matches!(class.methods[6].signature, MethodSignatureAst::SubscriptSetter { .. }));
        assert!(class.methods[7].is_constructor);
        assert!(class.methods[8].is_static);
        assert!(class.methods[9].is_foreign && class.methods[9].is_static);
    }

    #[test]
    fn parses_nested_string_interpolation() {
        let source = r#"var text = "answer %((1..3).map {|n| "%(n * n)"}.join())!""#;
        let module = parse_ok(source);
        let StmtKind::Variable {
            initializer: Some(expr),
            ..
        } = &module.statements[0].kind
        else {
            panic!("expected variable");
        };
        assert!(matches!(expr.kind, ExprKind::InterpolatedString(_)));
    }

    #[test]
    fn parses_block_arguments_without_fn_new() {
        let source = "numbers.map {|n| n * 2 }.where {|n| n < 100 }\n";
        let module = parse_ok(source);
        let StmtKind::Expression(expr) = &module.statements[0].kind else {
            panic!("expected expression statement");
        };
        let ExprKind::Invoke(where_call) = &expr.kind else {
            panic!("expected where call");
        };
        assert_eq!(where_call.name.value, "where");
        assert_eq!(where_call.args.len(), 1);
        assert!(matches!(&where_call.args[0].kind, ExprKind::Closure(_)));
    }

    #[test]
    fn parses_import_aliases() {
        let source = "import \"beverages\" for Coffee, Water as H2O, Tea\n";
        let module = parse_ok(source);
        let StmtKind::Import(import) = &module.statements[0].kind else {
            panic!("expected import");
        };
        assert_eq!(import.module, b"beverages");
        assert_eq!(import.bindings.len(), 3);
        assert_eq!(import.bindings[1].local.value, "H2O");
    }

    #[test]
    fn parses_attributes() {
        let source = r#"
#!doc = "runtime"
#meta(enabled = true, iterations = 32)
class Example {
  #!hot
  run() { 1 }
}
"#;
        let module = parse_ok(source);
        let StmtKind::Class(class) = &module.statements[0].kind else {
            panic!("expected class");
        };
        assert_eq!(class.attributes.len(), 2);
        assert!(class.attributes[0].runtime);
        assert_eq!(class.methods[0].attributes.len(), 1);
        assert!(class.methods[0].attributes[0].runtime);
    }

    #[test]
    fn raw_strings_trim_standalone_delimiter_lines() {
        let source = "var text = \"\"\"\n  hello\n  world\n\"\"\"\n";
        let module = parse_ok(source);
        let StmtKind::Variable {
            initializer: Some(expr),
            ..
        } = &module.statements[0].kind
        else {
            panic!("expected variable");
        };
        let ExprKind::String(bytes) = &expr.kind else {
            panic!("expected raw string");
        };
        assert_eq!(bytes, b"  hello\n  world");
    }

    #[test]
    fn supports_multiline_fluent_dot() {
        let source = "var result = numbers\n  .map {|n| n * 2 }\n  .toList\n";
        let module = parse_ok(source);
        assert_eq!(module.statements.len(), 1);
    }

    #[test]
    fn recovers_after_bad_statement() {
        let source = "var a =\nvar b = 2\nvar c = 3\n";
        let output = parse("test.wren", source);
        assert!(output.has_errors());
        assert!(output.module.statements.iter().any(|statement| {
            matches!(
                &statement.kind,
                StmtKind::Variable { name, .. } if name.value == "b" || name.value == "c"
            )
        }));
    }
}
