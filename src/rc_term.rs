use std::{
    cell::Cell,
    collections::HashMap,
    io::Write,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

#[derive(Clone, Debug)]
struct UnaryFnPayload {
    term: RcTerm,
    f: fn(f64) -> f64,
    grad: fn(f64) -> f64,
}

#[derive(Clone, Debug)]
enum TermInt {
    Value(Cell<f64>),
    Add(RcTerm, RcTerm),
    Sub(RcTerm, RcTerm),
    Mul(RcTerm, RcTerm),
    Div(RcTerm, RcTerm),
    UnaryFn(UnaryFnPayload),
}

impl TermInt {
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
struct TermPayload {
    name: String,
    value: TermInt,
    data: Cell<f64>,
    grad: Cell<f64>,
}

impl TermPayload {
    fn new(name: String, value: TermInt) -> TermPayload {
        let data = value.eval();
        Self {
            name,
            value,
            data: Cell::new(data),
            grad: Cell::new(0.),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RcTerm(Rc<TermPayload>);

impl Add for &RcTerm {
    type Output = RcTerm;
    fn add(self, rhs: Self) -> Self::Output {
        let name = format!("({} + {})", self.0.name, rhs.0.name);
        RcTerm::new_payload(TermPayload::new(
            name,
            TermInt::Add(self.clone(), rhs.clone()),
        ))
    }
}

impl Sub for &RcTerm {
    type Output = RcTerm;
    fn sub(self, rhs: Self) -> Self::Output {
        let name = format!("({} - {})", self.0.name, rhs.0.name);
        RcTerm::new_payload(TermPayload::new(
            name,
            TermInt::Sub(self.clone(), rhs.clone()),
        ))
    }
}

impl Mul for &RcTerm {
    type Output = RcTerm;
    fn mul(self, rhs: Self) -> Self::Output {
        let name = format!("{} * {}", self.0.name, rhs.0.name);
        RcTerm::new_payload(TermPayload::new(
            name,
            TermInt::Mul(self.clone(), rhs.clone()),
        ))
    }
}

impl Div for &RcTerm {
    type Output = RcTerm;
    fn div(self, rhs: Self) -> Self::Output {
        let name = format!("{} / {}", self.0.name, rhs.0.name);
        RcTerm::new_payload(TermPayload::new(
            name,
            TermInt::Div(self.clone(), rhs.clone()),
        ))
    }
}

impl RcTerm {
    pub fn new(name: impl Into<String>, val: f64) -> RcTerm {
        Self(Rc::new(TermPayload::new(
            name.into(),
            TermInt::Value(Cell::new(val)),
        )))
    }

    fn new_payload(val: TermPayload) -> Self {
        Self(Rc::new(val))
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
        let payload = &*self.0;
        payload as *const _ as usize
    }

    fn accum<'a>(&'a self, map: &mut HashMap<usize, (&'a TermPayload, Vec<usize>)>) {
        use TermInt::*;
        let parents = match &self.0.value {
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
        let grad = if self.id() == var.id() {
            1.
        } else {
            match &self.0.value {
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
        match &self.0.value {
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
        let grad = match &self.0.value {
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
        let val = match &self.0.value {
            Value(val) => val.get(),
            Add(lhs, rhs) => lhs.eval() + rhs.eval(),
            Sub(lhs, rhs) => lhs.eval() - rhs.eval(),
            Mul(lhs, rhs) => lhs.eval() * rhs.eval(),
            Div(lhs, rhs) => lhs.eval() / rhs.eval(),
            UnaryFn(UnaryFnPayload { term, f, .. }) => f(term.eval()),
        };
        self.0.data.set(val);
        val
    }

    pub fn exp(&self) -> Self {
        self.apply("exp", f64::exp, f64::exp)
    }

    pub fn apply(
        &self,
        name: &(impl AsRef<str> + ?Sized),
        f: fn(f64) -> f64,
        grad: fn(f64) -> f64,
    ) -> Self {
        let name = format!("{}({})", name.as_ref(), self.0.name.clone());
        Self::new_payload(TermPayload::new(
            name,
            TermInt::UnaryFn(UnaryFnPayload {
                term: self.clone(),
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
