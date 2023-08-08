//! Implementation of shared memory arena for the terms, aka a tape.
//! See https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

use std::{cell::RefCell, io::Write, rc::Rc};

use crate::{error::ValueNotDefinedError, tensor::Tensor};

#[derive(Default, Debug)]
/// A storage for [`TapeTerm`]s.
///
/// It is a growable buffer of expression nodes.
/// The implementation tend to be faster than nodes allocated randomly in heap memory.
/// Also the deallocation is much faster because it merely frees the dynamic array once.
pub struct Tape<T = f64> {
    nodes: RefCell<Vec<TapeNode<T>>>,
}

#[derive(Clone, Debug)]
pub struct TapeNode<T> {
    name: String,
    value: TapeValue<T>,
    data: Option<T>,
    grad: Option<T>,
}

#[derive(Clone)]
struct UnaryFnPayload<T> {
    term: u32,
    f: Rc<dyn Fn(T) -> T>,
    grad: Rc<dyn Fn(T) -> T>,
}

impl<T> std::fmt::Debug for UnaryFnPayload<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnaryFnPayload")
            .field("term", &self.term)
            .finish()
    }
}

#[derive(Clone)]
struct BinaryFnPayload<T> {
    lhs: u32,
    rhs: u32,
    f: Rc<dyn Fn(T, T) -> T>,
    /// In binary function, differentiation can be performed to both operands.
    grad: Rc<dyn Fn(T, T) -> (T, T)>,
}

impl<T> std::fmt::Debug for BinaryFnPayload<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinaryFnPayload")
            .field("lhs", &self.lhs)
            .field("rhs", &self.rhs)
            .finish()
    }
}

#[derive(Clone, Debug)]
enum TapeValue<T> {
    Value(T),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Div(u32, u32),
    Neg(u32),
    UnaryFn(UnaryFnPayload<T>),
    BinaryFn(BinaryFnPayload<T>),
    HStack(u32, u32),
}

/// An implementation of forward/reverse mode automatic differentiation, using memory arena called a [`Tape`],
/// to store term values.
/// It is more efficient than [`crate::RcTerm`], but you need to allocate a [`Tape`] before adding terms.
///
/// It accepts a value type `T` that implements [`Tensor`] trait.
///
/// # Example
///
/// ```
/// let tape = Tape::new();
/// let a = tape.term("a", 123.);
/// let b = tape.term("b", 321.);
/// let c = tape.term("c", 42.);
/// let ab = a + b;
/// let abc = ab * c;
/// println!("a + b = {}", ab.eval());
/// println!("(a + b) * c = {}", abc.eval());
/// let ab_a = ab.derive(&a);
/// println!("d(a + b) / da = {}", ab_a);
/// let abc_a = abc.derive(&a);
/// println!("d((a + b) * c) / da = {}", abc_a);
/// let abc_b = abc.derive(&b);
/// println!("d((a + b) * c) / db = {}", abc_b);
/// let abc_c = abc.derive(&c);
/// println!("d((a + b) * c) / dc = {}", abc_c);
///
/// let d = tape.term("d", 2.);
/// let abcd = abc / d;
/// let abcd_c = abcd.derive(&c);
/// println!("d((a + b) * c / d) / dc = {}", abcd_c);
/// ```
pub struct TapeTerm<'a, T = f64> {
    tape: &'a Tape<T>,
    idx: u32,
}

// derive macro doesn't work for generics
impl<'a, T> Clone for TapeTerm<'a, T> {
    fn clone(&self) -> Self {
        Self {
            tape: self.tape,
            idx: self.idx,
        }
    }
}
impl<'a, T> Copy for TapeTerm<'a, T> {}

impl<T: Tensor> Tape<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn term<'a>(&'a self, name: impl Into<String>, init: T) -> TapeTerm<'a, T> {
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(TapeNode {
            name: name.into(),
            value: TapeValue::Value(init),
            data: None,
            grad: None,
        });
        TapeTerm {
            tape: self,
            idx: idx as u32,
        }
    }

    fn term_name<'a>(&'a self, name: impl Into<String>, value: TapeValue<T>) -> TapeTerm<'a, T> {
        let mut nodes = self.nodes.borrow_mut();
        self.term_name_borrowed(&mut nodes, name, value)
    }

    fn term_name_borrowed<'a>(
        &'a self,
        nodes: &mut Vec<TapeNode<T>>,
        name: impl Into<String>,
        value: TapeValue<T>,
    ) -> TapeTerm<'a, T> {
        let idx = nodes.len();
        nodes.push(TapeNode {
            name: name.into(),
            value,
            data: None,
            grad: None,
        });
        TapeTerm {
            tape: self,
            idx: idx as u32,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }
}

impl<'a, T: Tensor> std::ops::Add for TapeTerm<'a, T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let name = {
            let nodes = self.tape.nodes.borrow();
            format!(
                "({} + {})",
                nodes[self.idx as usize].name, nodes[rhs.idx as usize].name
            )
        };
        self.tape.term_name(name, TapeValue::Add(self.idx, rhs.idx))
    }
}

impl<'a, T: Tensor> std::ops::Sub for TapeTerm<'a, T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let name = {
            let nodes = self.tape.nodes.borrow();
            format!(
                "({} - {})",
                nodes[self.idx as usize].name, nodes[rhs.idx as usize].name
            )
        };
        self.tape.term_name(name, TapeValue::Sub(self.idx, rhs.idx))
    }
}

impl<'a, T: Tensor> std::ops::Mul for TapeTerm<'a, T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let name = {
            let nodes = self.tape.nodes.borrow();
            format!(
                "{} * {}",
                nodes[self.idx as usize].name, nodes[rhs.idx as usize].name
            )
        };
        self.tape.term_name(name, TapeValue::Mul(self.idx, rhs.idx))
    }
}

impl<'a, T: Tensor> std::ops::Div for TapeTerm<'a, T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        let name = {
            let nodes = self.tape.nodes.borrow();
            format!(
                "{} / {}",
                nodes[self.idx as usize].name, nodes[rhs.idx as usize].name
            )
        };
        self.tape.term_name(name, TapeValue::Div(self.idx, rhs.idx))
    }
}

impl<'a, T: Tensor> std::ops::Neg for TapeTerm<'a, T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let name = {
            let nodes = self.tape.nodes.borrow();
            format!("-{}", nodes[self.idx as usize].name)
        };
        self.tape.term_name(name, TapeValue::Neg(self.idx))
    }
}

impl<'a, T: Tensor + 'static> TapeTerm<'a, T> {
    pub fn eval(&self) -> T {
        let mut nodes = self.tape.nodes.borrow_mut();
        clear(&mut nodes);
        let callback: Option<&fn(&[TapeNode<T>], u32)> = None;
        eval(&mut nodes, self.idx, callback)
    }

    pub fn eval_cb(&self, callback: &impl Fn(&[TapeNode<T>], u32)) -> T {
        let mut nodes = self.tape.nodes.borrow_mut();
        clear(&mut nodes);
        clear_grad(&mut nodes);
        eval(&mut nodes, self.idx, Some(callback))
    }

    pub fn value(&self) -> Option<T> {
        let nodes = self.tape.nodes.borrow();
        nodes[self.idx as usize].data.clone()
    }

    /// One-time derivation. Does not update internal gradient values.
    pub fn derive(&self, var: &Self) -> Option<T> {
        if self.idx == var.idx {
            Some(T::one())
        } else {
            let mut nodes = self.tape.nodes.borrow_mut();
            derive(&mut nodes, self.idx, var.idx)
        }
    }

    pub fn apply(
        &self,
        name: &(impl AsRef<str> + ?Sized),
        f: impl Fn(T) -> T + 'static,
        grad: impl Fn(T) -> T + 'static,
    ) -> Self {
        let self_name = self.tape.nodes.borrow()[self.idx as usize].name.clone();
        let name = format!("{}({})", name.as_ref(), self_name);
        self.tape.term_name(
            name,
            TapeValue::UnaryFn(UnaryFnPayload {
                term: self.idx,
                f: Rc::new(f),
                grad: Rc::new(grad),
            }),
        )
    }

    pub fn apply_binary(
        &self,
        name: &(impl AsRef<str> + ?Sized),
        rhs: TapeTerm<T>,
        f: impl Fn(T, T) -> T + 'static,
        grad: impl Fn(T, T) -> (T, T) + 'static,
    ) -> Self {
        let self_name = self.tape.nodes.borrow()[self.idx as usize].name.clone();
        let name = format!("{}({})", name.as_ref(), self_name);
        self.tape.term_name(
            name,
            TapeValue::BinaryFn(BinaryFnPayload {
                lhs: self.idx,
                rhs: rhs.idx,
                f: Rc::new(f),
                grad: Rc::new(grad),
            }),
        )
    }

    pub fn set(&self, value: T) -> Result<(), ()> {
        let mut nodes = self.tape.nodes.borrow_mut();
        let node = nodes.get_mut(self.idx as usize).ok_or_else(|| ())?;
        match &mut node.value {
            TapeValue::Value(val) => *val = value,
            _ => return Err(()),
        }
        Ok(())
    }

    pub fn backprop(&self) -> Result<(), ValueNotDefinedError> {
        let mut nodes = self.tape.nodes.borrow_mut();
        clear_grad(&mut nodes);
        backprop_rec(&mut nodes, self.idx, &|_, _| ())
    }

    pub fn backprop_cb(
        &self,
        callback: &impl Fn(&[TapeNode<T>], u32),
    ) -> Result<(), ValueNotDefinedError> {
        let mut nodes = self.tape.nodes.borrow_mut();
        clear_grad(&mut nodes);
        backprop_rec(&mut nodes, self.idx, callback)
    }

    /// Write graphviz dot file to the given writer.
    pub fn dot(&self, writer: &mut impl Write) -> std::io::Result<()> {
        self.dot_builder().dot(writer)
    }

    /// Create a builder for dot file writer configuration.
    pub fn dot_builder(&self) -> TapeDotBuilder<'a, T> {
        TapeDotBuilder {
            this: *self,
            show_values: false,
            vertical: false,
            hilight: None,
        }
    }

    pub fn grad(&self) -> Option<T> {
        self.tape.nodes.borrow()[self.idx as usize].grad.clone()
    }
}

fn clear<T: Tensor>(nodes: &mut [TapeNode<T>]) {
    for node in nodes {
        node.data = None;
    }
}

fn eval<T: Tensor + 'static>(
    nodes: &mut [TapeNode<T>],
    idx: u32,
    callback: Option<&impl Fn(&[TapeNode<T>], u32)>,
) -> T {
    use TapeValue::*;
    if let Some(ref data) = nodes[idx as usize].data {
        return data.clone();
    }
    let data = match &nodes[idx as usize].value {
        Value(val) => val.clone(),
        &Add(lhs, rhs) => eval(nodes, lhs, callback) + eval(nodes, rhs, callback),
        &Sub(lhs, rhs) => eval(nodes, lhs, callback) - eval(nodes, rhs, callback),
        &Mul(lhs, rhs) => eval(nodes, lhs, callback) * eval(nodes, rhs, callback),
        &Div(lhs, rhs) => eval(nodes, lhs, callback) / eval(nodes, rhs, callback),
        &Neg(term) => -eval(nodes, term, callback),
        &UnaryFn(UnaryFnPayload { term, ref f, .. }) => {
            let f = f.clone();
            let v = eval(nodes, term, callback);
            f(v)
        }
        &BinaryFn(BinaryFnPayload {
            lhs, rhs, ref f, ..
        }) => {
            let f = f.clone();
            let lhs = eval(nodes, lhs, callback);
            let rhs = eval(nodes, rhs, callback);
            f(lhs, rhs)
        }
        &HStack(lhs, rhs) => {
            let lhs = eval(nodes, lhs, callback);
            let rhs = eval(nodes, rhs, callback);
            lhs.hstack(rhs).unwrap()
        }
    };
    nodes[idx as usize].data = Some(data.clone());
    if let Some(callback) = callback {
        callback(nodes, idx);
    }
    data
}

pub trait HStack {
    fn hstack(self, rhs: Self) -> Self;
}

pub trait HSplit: Sized {
    fn hsplit(self, row: usize) -> (Self, Self);
}

impl<'a, T: Tensor + HStack> HStack for TapeTerm<'a, T> {
    fn hstack(self, rhs: Self) -> Self {
        let mut nodes = self.tape.nodes.borrow_mut();
        let lhs_name = &nodes[self.idx as usize].name;
        let rhs_name = &nodes[rhs.idx as usize].name;
        // let lhs_v = nodes[self.idx as usize].data.clone().unwrap();
        // let rhs_v = nodes[rhs.idx as usize].data.clone().unwrap();
        let name = format!("hstack({}, {})", lhs_name, rhs_name);
        self.tape
            .term_name_borrowed(&mut nodes, name, TapeValue::HStack(self.idx, rhs.idx))
    }
}

fn value<T: Clone>(nodes: &[TapeNode<T>], idx: u32) -> Option<T> {
    nodes[idx as usize].data.clone()
}

/// wrt - The variable to derive With Respect To
fn derive<T: Tensor>(nodes: &mut [TapeNode<T>], idx: u32, wrt: u32) -> Option<T> {
    use TapeValue::*;
    // println!("derive({}, {}): {:?}", idx, wrt, nodes[idx as usize].value);
    let grad = match nodes[idx as usize].value {
        Value(_) => {
            if idx == wrt {
                T::one()
            } else {
                T::default()
            }
        }
        Add(lhs, rhs) => derive(nodes, lhs, wrt)? + derive(nodes, rhs, wrt)?,
        Sub(lhs, rhs) => derive(nodes, lhs, wrt)? - derive(nodes, rhs, wrt)?,
        Mul(lhs, rhs) => {
            let dlhs = derive(nodes, lhs, wrt)?;
            let drhs = derive(nodes, rhs, wrt)?;
            dlhs * value(nodes, rhs)? + value(nodes, lhs)? * drhs
        }
        Div(lhs, rhs) => {
            let dlhs = derive(nodes, lhs, wrt)?;
            let drhs = derive(nodes, rhs, wrt)?;
            let elhs = value(nodes, lhs)?;
            let erhs = value(nodes, rhs)?;
            dlhs / erhs.clone() - elhs / erhs.clone() / erhs * drhs
        }
        Neg(term) => -derive(nodes, term, wrt)?,
        UnaryFn(UnaryFnPayload { term, ref grad, .. }) => {
            grad(value(nodes, term)?) * derive(nodes, term, wrt)?
        }
        BinaryFn(BinaryFnPayload {
            lhs, rhs, ref grad, ..
        }) => {
            let grad = grad.clone();
            let lhs_derive = derive(nodes, lhs, wrt)?;
            let rhs_derive = derive(nodes, rhs, wrt)?;
            let (lhs_grad, rhs_grad) = grad(value(nodes, lhs)?, value(nodes, rhs)?);
            lhs_grad * lhs_derive + rhs_grad * rhs_derive
        }
        HStack(lhs, rhs) => {
            let lhs = derive(nodes, lhs, wrt)?;
            let rhs = derive(nodes, rhs, wrt)?;
            lhs.hstack(rhs).unwrap()
        }
    };
    Some(grad)
}

fn clear_grad<T: Tensor>(nodes: &mut [TapeNode<T>]) {
    for node in nodes {
        node.grad = None;
    }
}

fn backprop_set<T: Tensor>(
    nodes: &mut [TapeNode<T>],
    idx: u32,
    grad: T,
    callback: &impl Fn(&[TapeNode<T>], u32),
) {
    if let Some(ref mut node_grad) = nodes[idx as usize].grad {
        *node_grad += grad.clone();
    } else {
        nodes[idx as usize].grad = Some(grad.clone());
    }
    callback(nodes, idx);
}

/// Assign gradient to all nodes
fn backprop_rec<T: Tensor>(
    nodes: &mut [TapeNode<T>],
    idx: u32,
    callback: &impl Fn(&[TapeNode<T>], u32),
) -> Result<(), ValueNotDefinedError> {
    use TapeValue::*;
    nodes[idx as usize].grad = Some(T::one());
    callback(nodes, idx);
    for i in (0..=idx).rev() {
        let grad = nodes[i as usize].grad.as_ref().unwrap().clone();
        match nodes[i as usize].value {
            Value(_) => (),
            Add(lhs, rhs) => {
                backprop_set(nodes, lhs, grad.clone(), callback);
                backprop_set(nodes, rhs, grad, callback);
            }
            Sub(lhs, rhs) => {
                backprop_set(nodes, lhs, grad.clone(), callback);
                backprop_set(nodes, rhs, -grad, callback);
            }
            Mul(lhs, rhs) => {
                let erhs = value(nodes, rhs).ok_or(ValueNotDefinedError)?;
                let elhs = value(nodes, lhs).ok_or(ValueNotDefinedError)?;
                backprop_set(nodes, lhs, grad.clone() * erhs, callback);
                backprop_set(nodes, rhs, grad * elhs, callback);
            }
            Div(lhs, rhs) => {
                let erhs = value(nodes, rhs).ok_or(ValueNotDefinedError)?;
                let elhs = value(nodes, lhs).ok_or(ValueNotDefinedError)?;
                backprop_set(nodes, lhs, grad.clone() / erhs.clone(), callback);
                backprop_set(nodes, rhs, -grad * elhs / erhs.clone() / erhs, callback);
            }
            Neg(term) => backprop_set(nodes, term, -grad, callback),
            UnaryFn(UnaryFnPayload {
                term, grad: ref g, ..
            }) => {
                let val = value(nodes, term).ok_or(ValueNotDefinedError)?;
                backprop_set(nodes, term, grad * g(val), callback)
            }
            BinaryFn(BinaryFnPayload {
                lhs,
                rhs,
                grad: ref g,
                ..
            }) => {
                let lhs_val = value(nodes, lhs).ok_or(ValueNotDefinedError)?;
                let rhs_val = value(nodes, rhs).ok_or(ValueNotDefinedError)?;
                let (lhs_grad, rhs_grad) = g(lhs_val, rhs_val);
                backprop_set(nodes, lhs, grad.clone() * lhs_grad, callback);
                backprop_set(nodes, rhs, grad * rhs_grad, callback);
            }
            HStack(lhs, rhs) => todo!(),
        }
    }
    Ok(())
}

/// The dot file writer configuration builder with the builder pattern.
pub struct TapeDotBuilder<'a, T: Default> {
    this: TapeTerm<'a, T>,
    show_values: bool,
    vertical: bool,
    hilight: Option<u32>,
}

impl<'a, T: Tensor> TapeDotBuilder<'a, T> {
    /// Set whether to show values and gradients of the terms on the node labels
    pub fn show_values(mut self, v: bool) -> Self {
        self.show_values = v;
        self
    }

    pub fn vertical(mut self, v: bool) -> Self {
        self.vertical = v;
        self
    }

    /// Set a term to show highlighted border around it.
    pub fn highlights(mut self, term: u32) -> Self {
        self.hilight = Some(term);
        self
    }

    /// Perform output of dot file
    pub fn dot(self, writer: &mut impl Write) -> std::io::Result<()> {
        let nodes = self.this.tape.nodes.borrow();
        self.dot_borrowed(&nodes, writer)
    }

    pub fn dot_borrowed(
        self,
        nodes: &[TapeNode<T>],
        writer: &mut impl Write,
    ) -> std::io::Result<()> {
        writeln!(
            writer,
            "digraph G {{\nrankdir=\"{}\";",
            if self.vertical { "TB" } else { "LR" }
        )?;
        for (id, term) in nodes.iter().enumerate() {
            let color = if term.grad.is_some() {
                "style=filled fillcolor=\"#ffff7f\""
            } else if term.data.is_some() {
                "style=filled fillcolor=\"#7fff7f\""
            } else {
                ""
            };
            let border = if self.hilight.is_some_and(|x| x == id as u32) {
                " color=red penwidth=2"
            } else {
                ""
            };
            let label = if self.show_values {
                format!(
                    "\\ndata:{}, grad:{}",
                    term.data
                        .as_ref()
                        .map(|v| format!("{v}"))
                        .unwrap_or_else(|| "None".into()),
                    term.grad
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
                id, term.name, label
            )?;
        }
        use TapeValue::*;
        for (id, term) in nodes.iter().enumerate() {
            let parents = match term.value {
                Value(_) => [None, None],
                Add(lhs, rhs) => [Some(lhs), Some(rhs)],
                Sub(lhs, rhs) => [Some(lhs), Some(rhs)],
                Mul(lhs, rhs) => [Some(lhs), Some(rhs)],
                Div(lhs, rhs) => [Some(lhs), Some(rhs)],
                Neg(term) => [Some(term), None],
                UnaryFn(UnaryFnPayload { term, .. }) => [Some(term), None],
                BinaryFn(BinaryFnPayload { lhs, rhs, .. }) => [Some(lhs), Some(rhs)],
                HStack(lhs, rhs) => [Some(lhs), Some(rhs)],
            };
            for pid in parents.into_iter().filter_map(|v| v) {
                writeln!(writer, "a{} -> a{};", pid, id)?;
            }
        }
        writeln!(writer, "}}")?;
        Ok(())
    }
}
