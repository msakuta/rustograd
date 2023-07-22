use std::{
    cell::Cell,
    ops::{Add, Div, Mul, Sub},
};

#[derive(Clone, Debug)]
struct UnaryFnPayload<'a> {
    term: &'a Term<'a>,
    f: fn(f64) -> f64,
    grad: fn(f64) -> f64,
}

#[derive(Clone, Debug)]
enum TermInt<'a> {
    Value(f64),
    Add(&'a Term<'a>, &'a Term<'a>),
    Sub(&'a Term<'a>, &'a Term<'a>),
    Mul(&'a Term<'a>, &'a Term<'a>),
    Div(&'a Term<'a>, &'a Term<'a>),
    UnaryFn(UnaryFnPayload<'a>),
}

#[derive(Clone, Debug)]
struct TermPayload<'a> {
    value: TermInt<'a>,
    grad: Cell<f64>,
}

impl<'a> TermPayload<'a> {
    fn new(value: TermInt<'a>) -> TermPayload<'a> {
        Self {
            value,
            grad: Cell::new(0.),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Term<'a>(Box<TermPayload<'a>>);

impl<'a> Add for &'a Term<'a> {
    type Output = Term<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        Term::new_payload(TermPayload::new(TermInt::Add(self, rhs)))
    }
}

impl<'a> Sub for &'a Term<'a> {
    type Output = Term<'a>;
    fn sub(self, rhs: Self) -> Self::Output {
        Term::new_payload(TermPayload::new(TermInt::Sub(self, rhs)))
    }
}

impl<'a> Mul for &'a Term<'a> {
    type Output = Term<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        Term::new_payload(TermPayload::new(TermInt::Mul(self, rhs)))
    }
}

impl<'a> Div for &'a Term<'a> {
    type Output = Term<'a>;
    fn div(self, rhs: Self) -> Self::Output {
        Term::new_payload(TermPayload::new(TermInt::Div(self, rhs)))
    }
}

impl<'a> Term<'a> {
    pub fn new(val: f64) -> Term<'a> {
        Self(Box::new(TermPayload::new(TermInt::Value(val))))
    }

    fn new_payload(val: TermPayload<'a>) -> Self {
        Self(Box::new(val))
    }

    pub fn grad(&self) -> f64 {
        self.0.grad.get()
    }

    /// One-time derivation. Does not update internal gradient values.
    pub fn derive(&self, var: &Self) -> f64 {
        use TermInt::*;
        let grad = if self as *const _ == var as *const _ {
            1.
        } else {
            match self.0.value {
                Value(_) => 0.,
                Add(lhs, rhs) => lhs.derive(var) + rhs.derive(var),
                Sub(lhs, rhs) => lhs.derive(var) - rhs.derive(var),
                Mul(lhs, rhs) => {
                    let dlhs = lhs.derive(var);
                    let drhs = rhs.derive(var);
                    dlhs * rhs.eval() + lhs.eval() * drhs
                }
                Div(lhs, rhs) => {
                    let dlhs = lhs.derive(var);
                    let drhs = rhs.derive(var);
                    if drhs == 0. {
                        dlhs / rhs.eval()
                    } else {
                        dlhs / rhs.eval() + lhs.eval() / drhs
                    }
                }
                UnaryFn(UnaryFnPayload { term, grad, .. }) => grad(var.eval()) * term.derive(var),
            }
        };
        self.0.grad.set(grad);
        grad
    }

    fn clear_grad(&self) {
        use TermInt::*;
        self.0.grad.set(0.);
        match self.0.value {
            Value(_) => (),
            Add(lhs, rhs) | Sub(lhs, rhs) | Mul(lhs, rhs) | Div(lhs, rhs) => {
                lhs.clear_grad();
                rhs.clear_grad();
            }
            UnaryFn(UnaryFnPayload { term, .. }) => term.clear_grad(),
        };
    }

    /// Assign gradient to all nodes
    fn backprop_rec(&self, grad: f64) -> f64 {
        use TermInt::*;
        self.0.grad.set(self.0.grad.get() + grad);
        let grad = match self.0.value {
            Value(_) => 0.,
            Add(lhs, rhs) => lhs.backprop_rec(grad) + rhs.backprop_rec(grad),
            Sub(lhs, rhs) => lhs.backprop_rec(grad) - rhs.backprop_rec(-grad),
            Mul(lhs, rhs) => {
                let dlhs = lhs.backprop_rec(rhs.eval());
                let drhs = rhs.backprop_rec(lhs.eval());
                dlhs * rhs.eval() + lhs.eval() * drhs
            }
            Div(lhs, rhs) => {
                let dlhs = lhs.backprop_rec(1. / rhs.eval());
                let drhs = rhs.backprop_rec(lhs.eval());
                if drhs == 0. {
                    dlhs / rhs.eval()
                } else {
                    dlhs / rhs.eval() + lhs.eval() / drhs
                }
            }
            UnaryFn(UnaryFnPayload { term, .. }) => term.backprop_rec(grad),
        };
        grad
    }

    /// The entry point to backpropagation
    pub fn backprop(&self) {
        self.clear_grad();
        self.backprop_rec(1.);
    }

    pub fn eval(&self) -> f64 {
        use TermInt::*;
        match self.0.value {
            Value(val) => val,
            Add(lhs, rhs) => lhs.eval() + rhs.eval(),
            Sub(lhs, rhs) => lhs.eval() - rhs.eval(),
            Mul(lhs, rhs) => lhs.eval() * rhs.eval(),
            Div(lhs, rhs) => lhs.eval() / rhs.eval(),
            UnaryFn(UnaryFnPayload { term, f, .. }) => f(term.eval()),
        }
    }

    pub fn exp(&'a self) -> Self {
        self.apply(f64::exp, f64::exp)
    }

    pub fn apply(&'a self, f: fn(f64) -> f64, grad: fn(f64) -> f64) -> Self {
        Self::new_payload(TermPayload::new(TermInt::UnaryFn(UnaryFnPayload {
            term: self,
            f,
            grad,
        })))
    }
}
