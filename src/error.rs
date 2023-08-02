use std::fmt::Display;

#[derive(Debug, Clone, Copy)]
pub struct ValueNotDefinedError;

impl Display for ValueNotDefinedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value is not defined in the forward pass. Call `eval()` before `backprop()`."
        )
    }
}
