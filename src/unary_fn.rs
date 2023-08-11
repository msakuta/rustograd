/// A trait that represents an unary operation on a value.
/// It needs to implement a transformation of the value, the gradient
/// and its reverse transformation (transposition).
pub trait UnaryFn<T> {
    fn name(&self) -> String;
    fn f(&self, data: T) -> T;
    fn grad(&self, data: T) -> T;
    fn t(&self, data: T) -> T;
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
