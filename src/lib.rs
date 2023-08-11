mod binary_fn;
mod dnum;
mod dnum_n2;
mod dvec;
pub mod error;
mod rc_term;
pub mod tape;
mod tensor;
mod term;
mod unary_fn;

pub use binary_fn::BinaryFn;
pub use dnum::Dnum;
pub use dnum_n2::Dnum2;
pub use dvec::Dvec;
pub use rc_term::{RcDotBuilder, RcTerm};
pub use tape::{Tape, TapeTerm};
pub use tensor::Tensor;
#[allow(deprecated)]
pub use term::Term;
pub use unary_fn::UnaryFn;
