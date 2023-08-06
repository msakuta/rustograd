pub mod error;
mod rc_term;
mod tape;
mod tensor;
mod term;

pub use rc_term::{RcDotBuilder, RcTerm};
pub use tape::{HSplit, HStack, Tape, TapeTerm};
pub use tensor::Tensor;
#[allow(deprecated)]
pub use term::Term;
