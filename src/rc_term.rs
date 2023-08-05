use std::{
    cell::RefCell,
    io::Write,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

use crate::{error::ValueNotDefinedError, tensor::Tensor};

#[derive(Clone)]
struct UnaryFnPayload<T: Tensor> {
    term: RcTerm<T>,
    f: fn(T) -> T,
    grad: fn(T) -> T,
}

enum TermInt<T: Tensor> {
    Value(RefCell<T>),
    Add(RcTerm<T>, RcTerm<T>),
    Sub(RcTerm<T>, RcTerm<T>),
    Mul(RcTerm<T>, RcTerm<T>),
    Div(RcTerm<T>, RcTerm<T>),
    Neg(RcTerm<T>),
    UnaryFn(UnaryFnPayload<T>),
}

impl<T: Tensor> TermInt<T> {
    fn eval(&self, callback: &impl Fn(RcTerm<T>)) -> T {
        use TermInt::*;
        match self {
            Value(val) => val.clone().into_inner(),
            Add(lhs, rhs) => lhs.eval_int(callback) + rhs.eval_int(callback),
            Sub(lhs, rhs) => lhs.eval_int(callback) - rhs.eval_int(callback),
            Mul(lhs, rhs) => lhs.eval_int(callback) * rhs.eval_int(callback),
            Div(lhs, rhs) => lhs.eval_int(callback) / rhs.eval_int(callback),
            Neg(term) => -term.eval_int(callback),
            UnaryFn(UnaryFnPayload { term, f, .. }) => f(term.eval_int(callback)),
        }
    }
}

struct TermPayload<T: Tensor> {
    name: String,
    value: TermInt<T>,
    data: RefCell<Option<T>>,
    grad: RefCell<Option<T>>,
}

impl<T: Tensor> TermPayload<T> {
    fn new(name: String, value: TermInt<T>) -> Self {
        let data = value.eval(&|_| ());
        Self {
            name,
            value,
            data: RefCell::new(Some(data)),
            grad: RefCell::new(None),
        }
    }
}

impl<T: Tensor> std::fmt::Debug for TermPayload<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TermPayload")
            .field("name", &self.name)
            .field("data", &self.data.borrow().is_some())
            .field("grad", &self.grad.borrow().is_some())
            .finish()
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
pub struct RcTerm<T: Tensor = f64>(Rc<TermPayload<T>>);

impl<T: Tensor> Add for &RcTerm<T> {
    type Output = RcTerm<T>;
    fn add(self, rhs: Self) -> Self::Output {
        let name = format!("({} + {})", self.0.name, rhs.0.name);
        RcTerm::new_payload(TermPayload::new(
            name,
            TermInt::Add(self.clone(), rhs.clone()),
        ))
    }
}

impl<T: Tensor> Sub for &RcTerm<T> {
    type Output = RcTerm<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        let name = format!("({} - {})", self.0.name, rhs.0.name);
        RcTerm::new_payload(TermPayload::new(
            name,
            TermInt::Sub(self.clone(), rhs.clone()),
        ))
    }
}

impl<T: Tensor> Mul for &RcTerm<T> {
    type Output = RcTerm<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        let name = format!("{} * {}", self.0.name, rhs.0.name);
        RcTerm::new_payload(TermPayload::new(
            name,
            TermInt::Mul(self.clone(), rhs.clone()),
        ))
    }
}

impl<T: Tensor> Div for &RcTerm<T> {
    type Output = RcTerm<T>;
    fn div(self, rhs: Self) -> Self::Output {
        let name = format!("{} / {}", self.0.name, rhs.0.name);
        RcTerm::new_payload(TermPayload::new(
            name,
            TermInt::Div(self.clone(), rhs.clone()),
        ))
    }
}

impl<T: Tensor> std::ops::Neg for &RcTerm<T> {
    type Output = RcTerm<T>;
    fn neg(self) -> Self::Output {
        let name = format!("-{}", self.0.name);
        RcTerm::new_payload(TermPayload::new(name, TermInt::Neg(self.clone())))
    }
}

impl<T: Tensor> RcTerm<T> {
    pub fn new(name: impl Into<String>, val: T) -> RcTerm<T> {
        Self(Rc::new(TermPayload::new(
            name.into(),
            TermInt::Value(RefCell::new(val)),
        )))
    }

    fn new_payload(val: TermPayload<T>) -> Self {
        Self(Rc::new(val))
    }

    pub fn grad(&self) -> T {
        self.0.grad.borrow().clone().unwrap()
    }

    /// Write graphviz dot file to the given writer.
    pub fn dot(&self, writer: &mut impl Write) -> std::io::Result<()> {
        self.dot_builder().dot(writer)
    }

    fn id(&self) -> usize {
        let payload = &*self.0;
        payload as *const _ as usize
    }

    fn accum<'a>(&'a self, map: &mut Vec<DotEntry<'a, T>>) {
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
    pub fn derive(&self, var: &Self) -> T {
        use TermInt::*;
        let grad = if self.id() == var.id() {
            T::one()
        } else {
            match &self.0.value {
                Value(_) => T::default(),
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
                    dlhs / erhs.clone() - elhs / erhs.clone() / erhs * drhs
                }
                Neg(term) => -term.derive(var),
                UnaryFn(UnaryFnPayload { term, grad, .. }) => grad(term.eval()) * term.derive(var),
            }
        };
        grad
    }

    pub fn clear_grad(&self) {
        use TermInt::*;
        *self.0.grad.borrow_mut() = None;
        match &self.0.value {
            Value(_) => (),
            Add(lhs, rhs) | Sub(lhs, rhs) | Mul(lhs, rhs) | Div(lhs, rhs) => {
                lhs.clear_grad();
                rhs.clear_grad();
            }
            Neg(term) | UnaryFn(UnaryFnPayload { term, .. }) => term.clear_grad(),
        };
    }

    fn backprop_accum(&self, list: &mut Vec<RcTerm<T>>) {
        use TermInt::*;
        match &self.0.value {
            Value(_) => (),
            Add(lhs, rhs) | Sub(lhs, rhs) | Mul(lhs, rhs) | Div(lhs, rhs) => {
                lhs.backprop_accum(list);
                rhs.backprop_accum(list);
            }
            Neg(term) | UnaryFn(UnaryFnPayload { term, .. }) => {
                term.backprop_accum(list);
            }
        }
        if !list.iter().any(|rc| rc.id() == self.id()) {
            list.push(self.clone());
        }
    }

    fn backprop_node(&self, grad: &T, callback: &impl Fn(RcTerm<T>)) {
        {
            let mut borrow = self.0.grad.borrow_mut();
            let grad_val =
                borrow.as_ref().map(Clone::clone).unwrap_or_else(T::default) + grad.clone();
            *borrow = Some(grad_val);
        }
        callback(self.clone());
    }

    /// Assign gradient to all nodes
    fn backprop_rec(
        list: &[RcTerm<T>],
        callback: &impl Fn(RcTerm<T>),
    ) -> Result<(), ValueNotDefinedError> {
        use TermInt::*;
        for node in list.iter().rev() {
            let null_callback = |_| ();
            let borrow = node.0.grad.borrow();
            let grad = borrow.as_ref().ok_or(ValueNotDefinedError)?;
            match &node.0.value {
                Value(_) => (),
                Add(lhs, rhs) => {
                    lhs.backprop_node(grad, callback);
                    rhs.backprop_node(grad, callback);
                }
                Sub(lhs, rhs) => {
                    lhs.backprop_node(grad, callback);
                    rhs.backprop_node(&-grad.clone(), callback);
                }
                Mul(lhs, rhs) => {
                    lhs.backprop_node(&(grad.clone() * rhs.eval_int(&null_callback)), callback);
                    rhs.backprop_node(&(grad.clone() * lhs.eval_int(&null_callback)), callback);
                }
                Div(lhs, rhs) => {
                    let erhs = rhs.eval_int(&null_callback);
                    let elhs = lhs.eval_int(&null_callback);
                    lhs.backprop_node(&(grad.clone() / erhs.clone()), callback);
                    rhs.backprop_node(&(-grad.clone() * elhs / erhs.clone() / erhs), callback);
                }
                Neg(term) => term.backprop_node(&-grad.clone(), callback),
                UnaryFn(UnaryFnPayload { term, grad: g, .. }) => {
                    let val = term.eval_int(&null_callback);
                    term.backprop_node(&(grad.clone() * g(val)), callback);
                }
            }
        }
        Ok(())
    }

    /// The entry point to backpropagation
    pub fn backprop(&self) -> Result<(), ValueNotDefinedError> {
        self.backprop_cb(&|_| ())
    }

    /// Backpropagation with a callback for each visited node
    pub fn backprop_cb(&self, callback: &impl Fn(RcTerm<T>)) -> Result<(), ValueNotDefinedError> {
        self.clear_grad();
        let mut list = vec![];
        self.backprop_accum(&mut list);
        self.backprop_node(&T::one(), callback);
        Self::backprop_rec(&list, callback)?;
        Ok(())
    }

    /// Evaluate value with possibly updated value by [`set`]
    pub fn eval(&self) -> T {
        self.clear();
        self.eval_int(&|_| ())
    }

    /// Evaluate value with a callback for each visited node
    pub fn eval_cb(&self, callback: &impl Fn(RcTerm<T>)) -> T {
        self.clear();
        self.eval_int(callback)
    }

    /// Internal function for recursive calls
    fn eval_int(&self, callback: &impl Fn(RcTerm<T>)) -> T {
        if let Some(data) = self.0.data.borrow().as_ref() {
            return data.clone();
        }
        let val = self.0.value.eval(callback);
        *self.0.data.borrow_mut() = Some(val.clone());
        if self.0.data.try_borrow().is_err() {
            println!("borrowed: true");
        }
        callback(self.clone());
        val
    }

    pub fn clear(&self) {
        use TermInt::*;
        *self.0.data.borrow_mut() = None;
        match &self.0.value {
            Value(_) => (),
            Add(lhs, rhs) | Sub(lhs, rhs) | Mul(lhs, rhs) | Div(lhs, rhs) => {
                lhs.clear();
                rhs.clear();
            }
            Neg(term) | UnaryFn(UnaryFnPayload { term, .. }) => term.clear(),
        };
    }

    // pub fn exp(&self) -> Self {
    //     self.apply("exp", f64::exp, f64::exp)
    // }

    pub fn apply(
        &self,
        name: &(impl AsRef<str> + ?Sized),
        f: fn(T) -> T,
        grad: fn(T) -> T,
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

    pub fn set(&self, value: T) -> Result<(), String> {
        if let TermInt::Value(ref rv) = self.0.value {
            *rv.borrow_mut() = value;
            Ok(())
        } else {
            Err("Cannot set value to non-leaf nodes".into())
        }
    }

    /// Create a builder for dot file writer configuration.
    pub fn dot_builder(&self) -> RcDotBuilder<T> {
        RcDotBuilder {
            this: self.clone(),
            show_values: true,
            hilight: None,
        }
    }
}

struct DotEntry<'a, T: Tensor> {
    id: usize,
    payload: &'a TermPayload<T>,
    parents: Vec<usize>,
}

/// The dot file writer configuration builder with the builder pattern.
pub struct RcDotBuilder<T: Tensor> {
    this: RcTerm<T>,
    show_values: bool,
    hilight: Option<RcTerm<T>>,
}

impl<T: Tensor> RcDotBuilder<T> {
    /// Set whether to show values and gradients of the terms on the node labels
    pub fn show_values(mut self, v: bool) -> Self {
        self.show_values = v;
        self
    }

    /// Set a term to show highlighted border around it.
    pub fn highlights(mut self, term: RcTerm<T>) -> Self {
        self.hilight = Some(term);
        self
    }

    /// Perform output of dot file
    pub fn dot(self, writer: &mut impl Write) -> std::io::Result<()> {
        let mut map = Vec::new();
        self.this.accum(&mut map);
        writeln!(writer, "digraph G {{\nrankdir=\"LR\";")?;
        for entry in &map {
            let DotEntry {
                id, payload: term, ..
            } = entry;
            let color = if term.grad.borrow().is_some() {
                "style=filled fillcolor=\"#ffff7f\""
            } else if term.data.borrow().is_some() {
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
            let label = if self.show_values {
                format!(
                    "\\ndata:{}, grad:{}",
                    term.data
                        .borrow()
                        .as_ref()
                        .map(|v| format!("{v}"))
                        .unwrap_or_else(|| "None".into()),
                    term.grad
                        .borrow()
                        .as_ref()
                        .map(|v| format!("{v:0.2}"))
                        .unwrap_or_else(|| "None".into())
                )
            } else {
                "".to_string()
            };
            writeln!(
                writer,
                "a{} [label=\"{}{}\" shape=rect {color}{border}];",
                *id, term.name, label
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
