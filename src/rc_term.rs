use std::{
    cell::Cell,
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
    fn eval(&self, callback: &impl Fn(RcTerm)) -> f64 {
        use TermInt::*;
        match self {
            Value(val) => val.get(),
            Add(lhs, rhs) => lhs.eval_int(callback) + rhs.eval_int(callback),
            Sub(lhs, rhs) => lhs.eval_int(callback) - rhs.eval_int(callback),
            Mul(lhs, rhs) => lhs.eval_int(callback) * rhs.eval_int(callback),
            Div(lhs, rhs) => lhs.eval_int(callback) / rhs.eval_int(callback),
            Neg(term) => -term.eval_int(callback),
            UnaryFn(UnaryFnPayload { term, f, .. }) => f(term.eval_int(callback)),
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
/// An implementation of forward/reverse mode automatic differentiation, using reference-counted box, [`Rc`],
/// to store term values.
/// It is more ergonomic to use, but tend to be slower than [`crate::TapeTerm`].
///
/// # Example
///
/// ```
/// let a = RcTerm::new("a", 123.);
/// let b = RcTerm::new("b", 321.);
/// let c = RcTerm::new("c", 42.);
/// let ab = &a + &b;
/// let abc = &ab * &c;
/// println!("a + b: {:?}", ab);
/// println!("(a + b) * c: {:?}", abc);
/// let ab_a = ab.derive(&a);
/// println!("d(a + b) / da = {:?}", ab_a);
/// let abc_a = abc.derive(&a);
/// println!("d((a + b) * c) / da = {}", abc_a);
/// let abc_b = abc.derive(&b);
/// println!("d((a + b) * c) / db = {}", abc_b);
/// let abc_c = abc.derive(&c);
/// println!("d((a + b) * c) / dc = {}", abc_c);
///
/// let d = RcTerm::new("d", 2.);
/// let abcd = &abc / &d;
/// let abcd_c = abcd.derive(&c);
/// println!("d((a + b) * c / d) / dc = {}", abcd_c);
///
/// abcd.backprop();
/// abcd.dot(&mut std::io::stdout()).unwrap();
/// ```
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
        self.dot_builder().dot(writer)
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
    fn backprop_rec(&self, grad: f64, callback: &impl Fn(RcTerm)) {
        use TermInt::*;
        let grad_val = self.0.grad.get().unwrap_or(0.) + grad;
        self.0.grad.set(Some(grad_val));
        callback(self.clone());
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
                lhs.backprop_rec(grad * rhs.eval_int(&null_callback), callback);
                rhs.backprop_rec(grad * lhs.eval_int(&null_callback), callback);
            }
            Div(lhs, rhs) => {
                let erhs = rhs.eval_int(&null_callback);
                let elhs = lhs.eval_int(&null_callback);
                lhs.backprop_rec(grad / erhs, callback);
                rhs.backprop_rec(-grad * elhs / erhs / erhs, callback);
            }
            Neg(term) => term.backprop_rec(-grad, callback),
            UnaryFn(UnaryFnPayload { term, grad: g, .. }) => {
                let val = term.eval_int(&null_callback);
                term.backprop_rec(grad * g(val), callback);
            }
        }
    }

    /// The entry point to backpropagation
    pub fn backprop(&self) {
        self.backprop_cb(&|_| ());
    }

    /// Backpropagation with a callback for each visited node
    pub fn backprop_cb(&self, callback: &impl Fn(RcTerm)) {
        self.clear_grad();
        self.backprop_rec(1., callback);
    }

    /// Evaluate value with possibly updated value by [`set`]
    pub fn eval(&self) -> f64 {
        self.clear();
        self.eval_int(&|_| ())
    }

    /// Evaluate value with a callback for each visited node
    pub fn eval_cb(&self, callback: &impl Fn(RcTerm)) -> f64 {
        self.clear();
        self.eval_int(callback)
    }

    /// Internal function for recursive calls
    fn eval_int(&self, callback: &impl Fn(RcTerm)) -> f64 {
        if let Some(data) = self.0.data.get() {
            return data;
        }
        let val = self.0.value.eval(callback);
        self.0.data.set(Some(val));
        callback(self.clone());
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

    pub fn dot_builder(&self) -> DotBuilder {
        DotBuilder {
            this: self.clone(),
            show_values: true,
            hilight: None,
        }
    }
}

struct DotEntry<'a> {
    id: usize,
    payload: &'a TermPayload,
    parents: Vec<usize>,
}

pub struct DotBuilder {
    this: RcTerm,
    show_values: bool,
    hilight: Option<RcTerm>,
}

impl DotBuilder {
    pub fn show_values(mut self, v: bool) -> DotBuilder {
        self.show_values = v;
        self
    }

    pub fn highlights(mut self, term: RcTerm) -> DotBuilder {
        self.hilight = Some(term);
        self
    }

    pub fn dot(self, writer: &mut impl Write) -> std::io::Result<()> {
        let mut map = Vec::new();
        self.this.accum(&mut map);
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
            let border = if self
                .hilight
                .as_ref()
                .is_some_and(|x| x.id() == *term as *const _ as usize)
            {
                " color=red penwidth=2"
            } else {
                ""
            };
            writeln!(
                writer,
                "a{} [label=\"{} \\ndata:{}, grad:{}\" shape=rect {color}{border}];",
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
}
