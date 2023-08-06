pub mod error;
mod rc_term;
mod tape;
mod tensor;
mod term;
mod dnum;

pub use dnum::Dnum;
pub use rc_term::{RcDotBuilder, RcTerm};
pub use tape::{Tape, TapeTerm};
pub use tensor::Tensor;
#[allow(deprecated)]
pub use term::Term;
