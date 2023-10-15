use std::fmt::Display;

#[derive(Debug, Clone, Copy)]
pub struct ValueNotDefinedError;

impl std::error::Error for ValueNotDefinedError {}

impl Display for ValueNotDefinedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value is not defined in the forward pass. Call `eval()` before `backprop()`."
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TermWasNotValueError;

impl std::error::Error for TermWasNotValueError {}

impl Display for TermWasNotValueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "The term being set a value is not of type Value. Setting a value is only allowed on the leaf of the expression graph."
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RangeError;

impl std::error::Error for RangeError {}

impl Display for RangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Accessing out of range of the tape nodes.")
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum RustogradError {
    ValueNotDefined(ValueNotDefinedError),
    TermWasNotValue(TermWasNotValueError),
    Range(RangeError),
}

impl std::error::Error for RustogradError {}

impl Display for RustogradError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ValueNotDefined(value) => value.fmt(f),
            Self::TermWasNotValue(value) => value.fmt(f),
            Self::Range(value) => value.fmt(f),
        }
    }
}

impl From<ValueNotDefinedError> for RustogradError {
    fn from(value: ValueNotDefinedError) -> Self {
        Self::ValueNotDefined(value)
    }
}

impl From<TermWasNotValueError> for RustogradError {
    fn from(value: TermWasNotValueError) -> Self {
        Self::TermWasNotValue(value)
    }
}

impl From<RangeError> for RustogradError {
    fn from(value: RangeError) -> Self {
        Self::Range(value)
    }
}
