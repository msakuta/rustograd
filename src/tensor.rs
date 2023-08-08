use std::fmt::Display;

use crate::HStack;

/// A trait that represents a type that can be used as a value in this library.
///
/// An implementation for f64 is provided by the crate, but you can implement it for
/// your custom type, such as vectors, matrices or complex numbers.
///
/// As a minimum-dependency library, this crate does not provide with the implementations
/// of other crate's tensor types, such as ndarray.
/// It also does not use popular trait libraries such as num-trait, so you need to implement
/// `one` and `is_zero` at least.
pub trait Tensor:
    std::ops::Add<Self, Output = Self>
    + std::ops::AddAssign<Self>
    + std::ops::Sub<Self, Output = Self>
    + std::ops::Mul<Self, Output = Self>
    + std::ops::Div<Self, Output = Self>
    + std::ops::Neg<Output = Self>
    + Sized
    + Default
    + Display
    + Clone
{
    fn one() -> Self;
    fn is_zero(&self) -> bool;
    fn hstack(self, _rhs: Self) -> Option<Self> {
        None
    }
}

impl Tensor for f64 {
    fn one() -> Self {
        1.
    }

    fn is_zero(&self) -> bool {
        *self == 0.
    }
}
