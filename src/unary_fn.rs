use crate::tape::{TapeIndex, TapeNode};

/// A trait that represents an unary operation on a value.
/// It needs to implement a transformation of the value, the gradient
/// and its reverse transformation (transposition).
pub trait UnaryFn<T> {
    fn name(&self) -> String;
    fn f(&self, data: T) -> T;
    fn grad(&self, data: T) -> T;
    fn t(&self, data: T) -> T {
        data
    }

    /// A method to generate a graph node that represents differentiation of this node.
    /// It takes 3 parameters:
    ///
    /// * `input` - a node that comes as an input variable of this node, e.g. x in exp(x).
    /// * `output` - a node that outputs the evaluated result, e.g. exp(x) itself in exp(x).
    /// * `derived` - a node representing a derived input, that is, x'.
    fn gen_graph(
        &self,
        _nodes: &mut Vec<TapeNode<T>>,
        _input: TapeIndex,
        _output: TapeIndex,
        _derived: TapeIndex,
    ) -> Option<TapeIndex> {
        None
    }
}

pub(crate) struct PtrUnaryFn<T> {
    pub name: String,
    pub f: fn(T) -> T,
    pub grad: fn(T) -> T,
}

impl<T> UnaryFn<T> for PtrUnaryFn<T> {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn f(&self, data: T) -> T {
        (self.f)(data)
    }
    fn grad(&self, data: T) -> T {
        (self.grad)(data)
    }
    fn t(&self, data: T) -> T {
        data
    }
}
