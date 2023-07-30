use std::{
    cell::Cell,
    collections::BTreeMap,
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
    Neg(RcTerm),
    UnaryFn(UnaryFnPayload),
}

impl TermInt {
    fn eval(&self, callback: &impl Fn(f64)) -> f64 {
        use TermInt::*;
        match self {
            Value(val) => val.get(),
            Add(lhs, rhs) => lhs.eval_cb(callback) + rhs.eval_cb(callback),
            Sub(lhs, rhs) => lhs.eval_cb(callback) - rhs.eval_cb(callback),
            Mul(lhs, rhs) => lhs.eval_cb(callback) * rhs.eval_cb(callback),
            Div(lhs, rhs) => lhs.eval_cb(callback) / rhs.eval_cb(callback),
            Neg(term) => -term.eval_cb(callback),
            UnaryFn(UnaryFnPayload { term, f, .. }) => f(term.eval_cb(callback)),
        }
    }
}

#[derive(Clone, Debug)]
struct TermPayload {
    name: String,
    value: TermInt,
    data: Cell<Option<f64>>,
    grad: Cell<Option<f64>>,
}

impl TermPayload {
    fn new(name: String, value: TermInt) -> TermPayload {
        let data = value.eval(&|_| ());
        Self {
            name,
            value,
            data: Cell::new(Some(data)),
            grad: Cell::new(None),
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

impl<'a> std::ops::Neg for &RcTerm {
    type Output = RcTerm;
    fn neg(self) -> Self::Output {
        let name = format!("-{}", self.0.name);
        RcTerm::new_payload(TermPayload::new(name, TermInt::Neg(self.clone())))
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
        self.0.grad.get().unwrap()
    }

    /// Write graphviz dot file to the given writer.
    pub fn dot(&self, writer: &mut impl Write) -> std::io::Result<()> {
        let mut map = Vec::new();
        self.accum(&mut map);
        writeln!(writer, "digraph G {{\nrankdir=\"LR\";")?;
        for entry in &map {
            let DotEntry {
                id, payload: term, ..
            } = entry;
            let color = if term.grad.get().is_some() {
                "style=filled fillcolor=\"#ffff7f\""
            } else if term.data.get().is_some() {
                "style=filled fillcolor=\"#7fff7f\""
            } else {
                ""
            };
            writeln!(
                writer,
                "a{} [label=\"{} \\ndata:{}, grad:{}\" shape=rect {color}];",
                *id,
                term.name,
                term.data
                    .get()
                    .map(|v| format!("{v}"))
                    .unwrap_or_else(|| "None".into()),
                term.grad
                    .get()
                    .map(|v| format!("{v:0.2}"))
                    .unwrap_or_else(|| "None".into())
            )?;
        }
        for entry in &map {
            let DotEntry { id, parents, .. } = entry;
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

    fn accum<'a>(&'a self, map: &mut Vec<DotEntry<'a>>) {
        use TermInt::*;
        let parents = match &self.0.value {
            Value(_) => vec![],
            Add(lhs, rhs) | Sub(lhs, rhs) | Mul(lhs, rhs) | Div(lhs, rhs) => {
                lhs.accum(map);
                rhs.accum(map);
                vec![lhs.id(), rhs.id()]
            }
            Neg(term) | UnaryFn(UnaryFnPayload { term, .. }) => {
                term.accum(map);
                vec![term.id()]
            }
        };
        map.push(DotEntry {
            id: self.id(),
            parents,
            payload: &self.0,
        });
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
                    let elhs = lhs.eval();
                    let erhs = rhs.eval();
                    let dlhs = lhs.derive(var);
                    let drhs = rhs.derive(var);
                    dlhs / erhs - elhs / erhs / erhs * drhs
                }
                Neg(term) => -term.derive(var),
                UnaryFn(UnaryFnPayload { term, grad, .. }) => grad(term.eval()) * term.derive(var),
            }
        };
        grad
    }

    pub fn clear_grad(&self) {
        use TermInt::*;
        self.0.grad.set(None);
        match &self.0.value {
            Value(_) => (),
            Add(lhs, rhs) | Sub(lhs, rhs) | Mul(lhs, rhs) | Div(lhs, rhs) => {
                lhs.clear_grad();
                rhs.clear_grad();
            }
            Neg(term) | UnaryFn(UnaryFnPayload { term, .. }) => term.clear_grad(),
        };
    }

    /// Assign gradient to all nodes
    fn backprop_rec(&self, grad: f64, callback: &impl Fn(f64)) {
        use TermInt::*;
        let grad_val = self.0.grad.get().unwrap_or(0.) + grad;
        self.0.grad.set(Some(grad_val));
        callback(grad_val);
        let null_callback = |_| ();
        match &self.0.value {
            Value(_) => (),
            Add(lhs, rhs) => {
                lhs.backprop_rec(grad, callback);
                rhs.backprop_rec(grad, callback);
            }
            Sub(lhs, rhs) => {
                lhs.backprop_rec(grad, callback);
                rhs.backprop_rec(-grad, callback);
            }
            Mul(lhs, rhs) => {
                lhs.backprop_rec(grad * rhs.eval_cb(&null_callback), callback);
                rhs.backprop_rec(grad * lhs.eval_cb(&null_callback), callback);
            }
            Div(lhs, rhs) => {
                let erhs = rhs.eval_cb(&null_callback);
                let elhs = lhs.eval_cb(&null_callback);
                lhs.backprop_rec(grad / erhs, callback);
                rhs.backprop_rec(-grad * elhs / erhs / erhs, callback);
            }
            Neg(term) => term.backprop_rec(-grad, callback),
            UnaryFn(UnaryFnPayload { term, grad: g, .. }) => {
                let val = term.eval_cb(&null_callback);
                term.backprop_rec(grad * g(val), callback);
            }
        }
    }

    /// The entry point to backpropagation
    pub fn backprop(&self) {
        self.backprop_cb(&|_| ());
    }

    /// Backpropagation with a callback for each visited node
    pub fn backprop_cb(&self, callback: &impl Fn(f64)) {
        self.clear_grad();
        self.backprop_rec(1., callback);
    }

    /// Evaluate value with possibly updated value by [`set`]
    pub fn eval(&self) -> f64 {
        self.eval_cb(&|_| ())
    }

    /// Evaluate value with a callback for each visited node
    pub fn eval_cb(&self, callback: &impl Fn(f64)) -> f64 {
        let val = self.0.value.eval(callback);
        self.0.data.set(Some(val));
        callback(val);
        val
    }

    pub fn clear(&self) {
        use TermInt::*;
        self.0.data.set(None);
        match &self.0.value {
            Value(_) => (),
            Add(lhs, rhs) | Sub(lhs, rhs) | Mul(lhs, rhs) | Div(lhs, rhs) => {
                lhs.clear();
                rhs.clear();
            }
            Neg(term) | UnaryFn(UnaryFnPayload { term, .. }) => term.clear(),
        };
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

struct DotEntry<'a> {
    id: usize,
    payload: &'a TermPayload,
    parents: Vec<usize>,
}
