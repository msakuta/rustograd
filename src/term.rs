#![allow(deprecated)]
use std::{
    cell::Cell,
    collections::HashMap,
    io::Write,
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
    Value(Cell<f64>),
    Add(&'a Term<'a>, &'a Term<'a>),
    Sub(&'a Term<'a>, &'a Term<'a>),
    Mul(&'a Term<'a>, &'a Term<'a>),
    Div(&'a Term<'a>, &'a Term<'a>),
    UnaryFn(UnaryFnPayload<'a>),
}

impl<'a> TermInt<'a> {
    fn eval(&self) -> f64 {
        use TermInt::*;
        match self {
            Value(val) => val.get(),
            Add(lhs, rhs) => lhs.eval() + rhs.eval(),
            Sub(lhs, rhs) => lhs.eval() - rhs.eval(),
            Mul(lhs, rhs) => lhs.eval() * rhs.eval(),
            Div(lhs, rhs) => lhs.eval() / rhs.eval(),
            UnaryFn(UnaryFnPayload { term, f, .. }) => f(term.eval()),
        }
    }
}

#[derive(Clone, Debug)]
struct TermPayload<'a> {
    name: String,
    value: TermInt<'a>,
    data: Cell<f64>,
    grad: Cell<f64>,
}

impl<'a> TermPayload<'a> {
    fn new(name: String, value: TermInt<'a>) -> TermPayload<'a> {
        let data = value.eval();
        Self {
            name,
            value,
            data: Cell::new(data),
            grad: Cell::new(0.),
        }
    }
}

#[deprecated]
#[derive(Clone, Debug)]
/// An implementation of forward/reverse mode automatic differentiation, deprecated in favor of [`crate::TapeTerm`],
/// which uses more compact memory representation and more ergonomic to use.
///
/// # Example
///
/// ```
/// let a = Term::new("a", 123.);
/// let b = Term::new("b", 321.);
/// let c = Term::new("c", 42.);
/// let ab = &a + &b;
/// let abc = &ab * &c;
/// let abc_a = abc.derive(&a);
/// println!("d((a + b) * c) / da = {}", abc_a); // 42
/// let abc_b = abc.derive(&b);
/// println!("d((a + b) * c) / db = {}", abc_b); // 42
/// let abc_c = abc.derive(&c);
/// println!("d((a + b) * c) / dc = {}", abc_c); // 444
///
/// let d = Term::new("d", 2.);
/// let abcd = &abc / &d;
/// let abcd_c = abcd.derive(&c);
/// println!("d((a + b) * c / d) / dc = {}", abcd_c);
///
/// abc.backprop();
/// abc.dot(&mut std::io::stdout()).unwrap();
/// ```
pub struct Term<'a>(Box<TermPayload<'a>>);

impl<'a> Add for &'a Term<'a> {
    type Output = Term<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        let name = format!("({} + {})", self.0.name, rhs.0.name);
        Term::new_payload(TermPayload::new(name, TermInt::Add(self, rhs)))
    }
}

impl<'a> Sub for &'a Term<'a> {
    type Output = Term<'a>;
    fn sub(self, rhs: Self) -> Self::Output {
        let name = format!("({} - {})", self.0.name, rhs.0.name);
        Term::new_payload(TermPayload::new(name, TermInt::Sub(self, rhs)))
    }
}

impl<'a> Mul for &'a Term<'a> {
    type Output = Term<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        let name = format!("{} * {}", self.0.name, rhs.0.name);
        Term::new_payload(TermPayload::new(name, TermInt::Mul(self, rhs)))
    }
}

impl<'a> Div for &'a Term<'a> {
    type Output = Term<'a>;
    fn div(self, rhs: Self) -> Self::Output {
        let name = format!("{} / {}", self.0.name, rhs.0.name);
        Term::new_payload(TermPayload::new(name, TermInt::Div(self, rhs)))
    }
}

impl<'a> Term<'a> {
    pub fn new(name: impl Into<String>, val: f64) -> Term<'a> {
        Self(Box::new(TermPayload::new(
            name.into(),
            TermInt::Value(Cell::new(val)),
        )))
    }

    fn new_payload(val: TermPayload<'a>) -> Self {
        Self(Box::new(val))
    }

    pub fn grad(&self) -> f64 {
        self.0.grad.get()
    }

    /// Write graphviz dot file to the given writer.
    pub fn dot(&self, writer: &mut impl Write) -> std::io::Result<()> {
        let mut map = HashMap::new();
        self.accum(&mut map);
        writeln!(writer, "digraph G {{\nrankdir=\"LR\";")?;
        for (id, (term, _)) in &map {
            writeln!(
                writer,
                "a{} [label=\"{} \\ndata:{}, grad:{}\"];",
                *id,
                term.name,
                term.data.get(),
                term.grad.get()
            )?;
        }
        for (id, (_, parents)) in &map {
            for pid in parents {
                writeln!(writer, "a{} -> a{};", pid, *id)?;
            }
        }
        writeln!(writer, "}}")?;
        Ok(())
    }

    fn id(&self) -> usize {
        self as *const _ as usize
    }

    fn accum(&'a self, map: &mut HashMap<usize, (&'a TermPayload<'a>, Vec<usize>)>) {
        use TermInt::*;
        let parents = match self.0.value {
            Value(_) => vec![],
            Add(lhs, rhs) | Sub(lhs, rhs) | Mul(lhs, rhs) | Div(lhs, rhs) => {
                lhs.accum(map);
                rhs.accum(map);
                vec![lhs.id(), rhs.id()]
            }
            UnaryFn(UnaryFnPayload { term, .. }) => {
                term.accum(map);
                vec![term.id()]
            }
        };
        map.insert(self.id(), (&self.0, parents));
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
                UnaryFn(UnaryFnPayload { term, grad, .. }) => grad(term.eval()) * term.derive(var),
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
            UnaryFn(UnaryFnPayload { term, grad: g, .. }) => term.backprop_rec(g(grad)),
        };
        grad
    }

    /// The entry point to backpropagation
    pub fn backprop(&self) {
        self.clear_grad();
        self.backprop_rec(1.);
    }

    pub fn eval(&self) -> f64 {
        let val = self.0.value.eval();
        self.0.data.set(val);
        val
    }

    pub fn exp(&'a self) -> Self {
        self.apply("exp", f64::exp, f64::exp)
    }

    pub fn apply(
        &'a self,
        name: &(impl AsRef<str> + ?Sized),
        f: fn(f64) -> f64,
        grad: fn(f64) -> f64,
    ) -> Self {
        let name = format!("{}({})", name.as_ref(), self.0.name.clone());
        Self::new_payload(TermPayload::new(
            name,
            TermInt::UnaryFn(UnaryFnPayload {
                term: self,
                f,
                grad,
            }),
        ))
    }

    pub fn set(&self, value: f64) -> Result<(), String> {
        if let TermInt::Value(ref rv) = self.0.value {
            rv.set(value);
            Ok(())
        } else {
            Err("Cannot set value to non-leaf nodes".into())
        }
    }
}
