mod rc_term;
mod tape;
mod term;

pub use rc_term::{RcDotBuilder, RcTerm};
pub use tape::{Tape, TapeTerm, Tensor};
#[allow(deprecated)]
pub use term::Term;
