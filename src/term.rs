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
    Value(f64),
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
            Value(val) => *val,
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
    data: f64,
    grad: Cell<f64>,
}

impl<'a> TermPayload<'a> {
    fn new(name: String, value: TermInt<'a>) -> TermPayload<'a> {
        let data = value.eval();
        Self {
            name,
            value,
            data,
            grad: Cell::new(0.),
        }
    }
}

#[derive(Clone, Debug)]
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
        Self(Box::new(TermPayload::new(name.into(), TermInt::Value(val))))
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
                term.data,
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
}
