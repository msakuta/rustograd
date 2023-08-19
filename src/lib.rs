mod dnum;
mod dvec;
pub mod error;
mod rc_term;
pub mod tape;
mod tensor;
mod term;
mod unary_fn;

pub use dnum::Dnum;
pub use dvec::Dvec;
pub use rc_term::{RcDotBuilder, RcTerm};
pub use tape::{Tape, TapeTerm};
pub use tensor::Tensor;
#[allow(deprecated)]
pub use term::Term;
pub use unary_fn::UnaryFn;
