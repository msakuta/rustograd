mod dnum;
mod dvec;
pub mod error;
mod rc_term;
mod tape;
mod tensor;
mod term;

pub use dnum::Dnum;
pub use dvec::Dvec;
pub use rc_term::{RcDotBuilder, RcTerm};
pub use tape::{Tape, TapeFn, TapeTerm};
pub use tensor::Tensor;
#[allow(deprecated)]
pub use term::Term;
