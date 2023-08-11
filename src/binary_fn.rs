/// A trait that represents a binary operation on a value.
/// It needs to implement a transformation of the value, the gradient
/// and its reverse transformation (transposition).
pub trait BinaryFn<T> {
    fn name(&self) -> String;
    fn f(&self, lhs: T, rhs: T) -> T;
    fn grad(&self, lhs: T, rhs: T) -> T;
    fn t(&self, data: T) -> (T, T);
}

pub(crate) struct PtrBinaryFn<T> {
    pub name: String,
    pub f: fn(T, T) -> T,
    pub grad: fn(T, T) -> T,
    pub t: fn(T) -> (T, T),
}

impl<T> BinaryFn<T> for PtrBinaryFn<T> {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn f(&self, lhs: T, rhs: T) -> T {
        (self.f)(lhs, rhs)
    }
    fn grad(&self, lhs: T, rhs: T) -> T {
        (self.grad)(lhs, rhs)
    }
    fn t(&self, data: T) -> (T, T) {
        (self.t)(data)
    }
}
