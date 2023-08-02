//! Implementation of shared memory arena for the terms, aka a tape.
//! See https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

use std::{cell::RefCell, fmt::Display, io::Write};

use crate::error::ValueNotDefinedError;

/// A trait that represents a type that can be used as a value in this library.
///
/// An implementation for f64 is provided by the crate, but you can implement it for
/// your custom type, such as vectors, matrices or complex numbers.
///
/// As a minimum-dependency library, this crate does not provide with the implementations
/// of other crate's tensor types, such as ndarray.
/// It also does not use popular trait libraries such as num-trait, so you need to implement
/// `one` and `is_zero` at least.
pub trait Tensor:
    std::ops::Add<Self, Output = Self>
    + std::ops::AddAssign<Self>
    + std::ops::Sub<Self, Output = Self>
    + std::ops::Mul<Self, Output = Self>
    + std::ops::Div<Self, Output = Self>
    + std::ops::Neg<Output = Self>
    + Sized
    + Default
    + Display
    + Clone
{
    fn one() -> Self;
    fn is_zero(&self) -> bool;
}

impl Tensor for f64 {
    fn one() -> Self {
        1.
    }

    fn is_zero(&self) -> bool {
        *self == 0.
    }
}

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

#[derive(Clone, Debug)]
struct UnaryFnPayload<T> {
    term: u32,
    f: fn(T) -> T,
    grad: fn(T) -> T,
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
        eval(&mut nodes, self.idx, Some(callback))
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
        f: fn(T) -> T,
        grad: fn(T) -> T,
    ) -> Self {
        let self_name = self.tape.nodes.borrow()[self.idx as usize].name.clone();
        let name = format!("{}({})", name.as_ref(), self_name);
        self.tape.term_name(
            name,
            TapeValue::UnaryFn(UnaryFnPayload {
                term: self.idx,
                f,
                grad,
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
        let callback: Option<&fn(&[TapeNode<T>], u32)> = None;
        backprop_rec(&mut nodes, self.idx, T::one(), callback)
    }

    pub fn backprop_cb(
        &self,
        callback: &impl Fn(&[TapeNode<T>], u32),
    ) -> Result<(), ValueNotDefinedError> {
        let mut nodes = self.tape.nodes.borrow_mut();
        clear_grad(&mut nodes);
        backprop_rec(&mut nodes, self.idx, T::one(), Some(callback))
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
        &UnaryFn(UnaryFnPayload { term, f, .. }) => f(eval(nodes, term, callback)),
    };
    nodes[idx as usize].data = Some(data.clone());
    if let Some(callback) = callback {
        callback(nodes, idx);
    }
    data
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
        UnaryFn(UnaryFnPayload { term, grad, .. }) => {
            grad(value(nodes, term)?) * derive(nodes, term, wrt)?
        }
    };
    Some(grad)
}

fn clear_grad<T: Tensor>(nodes: &mut [TapeNode<T>]) {
    for node in nodes {
        node.grad = None;
    }
}

/// Assign gradient to all nodes
fn backprop_rec<T: Tensor>(
    nodes: &mut [TapeNode<T>],
    idx: u32,
    grad: T,
    callback: Option<&impl Fn(&[TapeNode<T>], u32)>,
) -> Result<(), ValueNotDefinedError> {
    use TapeValue::*;
    if let Some(ref mut node_grad) = nodes[idx as usize].grad {
        *node_grad += grad.clone();
    } else {
        nodes[idx as usize].grad = Some(grad.clone());
    }
    if let Some(callback) = callback {
        callback(nodes, idx);
    }
    match nodes[idx as usize].value {
        Value(_) => (),
        Add(lhs, rhs) => {
            backprop_rec(nodes, lhs, grad.clone(), callback)?;
            backprop_rec(nodes, rhs, grad.clone(), callback)?;
        }
        Sub(lhs, rhs) => {
            backprop_rec(nodes, lhs, grad.clone(), callback)?;
            backprop_rec(nodes, rhs, -grad, callback)?;
        }
        Mul(lhs, rhs) => {
            let erhs = value(nodes, rhs).ok_or(ValueNotDefinedError)?;
            let elhs = value(nodes, lhs).ok_or(ValueNotDefinedError)?;
            backprop_rec(nodes, lhs, grad.clone() * erhs, callback)?;
            backprop_rec(nodes, rhs, grad * elhs, callback)?;
        }
        Div(lhs, rhs) => {
            let erhs = value(nodes, rhs).ok_or(ValueNotDefinedError)?;
            let elhs = value(nodes, lhs).ok_or(ValueNotDefinedError)?;
            backprop_rec(nodes, lhs, grad.clone() / erhs.clone(), callback)?;
            backprop_rec(nodes, rhs, -grad * elhs / erhs.clone() / erhs, callback)?;
        }
        Neg(term) => backprop_rec(nodes, term, -grad, callback)?,
        UnaryFn(UnaryFnPayload { term, grad: g, .. }) => {
            let val = value(nodes, term).ok_or(ValueNotDefinedError)?;
            backprop_rec(nodes, term, grad * g(val), callback)?
        }
    }
    Ok(())
}

/// The dot file writer configuration builder with the builder pattern.
pub struct TapeDotBuilder<'a, T: Default> {
    this: TapeTerm<'a, T>,
    show_values: bool,
    hilight: Option<u32>,
}

impl<'a, T: Tensor> TapeDotBuilder<'a, T> {
    /// Set whether to show values and gradients of the terms on the node labels
    pub fn show_values(mut self, v: bool) -> Self {
        self.show_values = v;
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
        writeln!(writer, "digraph G {{\nrankdir=\"LR\";")?;
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
            };
            for pid in parents.into_iter().filter_map(|v| v) {
                writeln!(writer, "a{} -> a{};", pid, id)?;
            }
        }
        writeln!(writer, "}}")?;
        Ok(())
    }
}
