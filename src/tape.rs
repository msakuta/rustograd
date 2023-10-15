//! Implementation of shared memory arena for the terms, aka a tape.
//! See https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

use std::{cell::RefCell, io::Write};

use crate::{
    error::{RangeError, RustogradError, TermWasNotValueError, ValueNotDefinedError},
    tensor::Tensor,
    unary_fn::{PtrUnaryFn, UnaryFn},
    BinaryFn,
};

#[derive(Default, Debug)]
/// A storage for [`TapeTerm`]s.
///
/// It is a growable buffer of expression nodes.
/// The implementation tend to be faster than nodes allocated randomly in heap memory.
/// Also the deallocation is much faster because it merely frees the dynamic array once.
pub struct Tape<T = f64> {
    nodes: RefCell<Vec<TapeNode<T>>>,
}

pub type TapeIndex = u32;
pub const TAPE_ZERO: TapeIndex = 0;
pub const TAPE_ONE: TapeIndex = 1;

#[derive(Debug)]
pub struct TapeNode<T> {
    name: String,
    value: TapeValue<T>,
    data: Option<T>,
    grad: Option<T>,
}

struct UnaryFnPayload<T> {
    term: TapeIndex,
    f: Option<Box<dyn UnaryFn<T>>>,
}

impl<T> std::fmt::Debug for UnaryFnPayload<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TermPayload")
            .field("term", &self.term)
            .field("f", &"<dyn TapeFn>")
            .finish()
    }
}

struct BinaryFnPayload<T> {
    lhs: u32,
    rhs: u32,
    f: Option<Box<dyn BinaryFn<T>>>,
}

impl<T> std::fmt::Debug for BinaryFnPayload<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TermPayload")
            .field("lhs", &self.lhs)
            .field("rhs", &self.rhs)
            .field("f", &"<dyn TapeFn>")
            .finish()
    }
}

#[derive(Debug)]
enum TapeValue<T> {
    Value(T),
    Add(TapeIndex, TapeIndex),
    Sub(TapeIndex, TapeIndex),
    Mul(TapeIndex, TapeIndex),
    Div(TapeIndex, TapeIndex),
    Neg(TapeIndex),
    UnaryFn(UnaryFnPayload<T>),
    BinaryFn(BinaryFnPayload<T>),
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
/// let tape = rustograd::Tape::new();
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
    idx: TapeIndex,
}

impl<'a, T> TapeTerm<'a, T> {
    pub fn to_tape_index(&self) -> TapeIndex {
        self.idx
    }
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
        // The first and the second entries in a tape are reserved for 0 (additive identity)
        // and 1 (multiplicative identity) to make generating graph more efficiently.
        // We pay 2 elements worth of allocation for every tape in exchange, but I think
        // it is a fair trade, because autograd is usually used with complex expression
        // (if your expression is simple enough that just 2 pre-allocaed nodes can be
        // an overhead, why would you need autograd in the first place?)
        Self {
            nodes: RefCell::new(vec![
                TapeNode {
                    name: "0".to_string(),
                    value: TapeValue::Value(T::default()),
                    data: Some(T::default()),
                    grad: Some(T::default()),
                },
                TapeNode {
                    name: "1".to_string(),
                    value: TapeValue::Value(T::one()),
                    data: Some(T::one()),
                    grad: Some(T::default()),
                },
            ]),
        }
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
            idx: idx as TapeIndex,
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
            idx: idx as TapeIndex,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    pub fn zero(&self) -> TapeTerm<T> {
        // Be sure to allocate zero in new()
        TapeTerm { tape: self, idx: 0 }
    }

    pub fn one(&self) -> TapeTerm<T> {
        // Be sure to allocate one in new()
        TapeTerm { tape: self, idx: 1 }
    }
}

#[test]
fn test_zero_one() {
    let tape = Tape::<f64>::new();
    let zero = tape.zero();
    let one = tape.one();
    assert_eq!((zero + one).eval(), 1.);
    assert_eq!((zero * one).eval(), 0.);
}

impl<T: Tensor + std::fmt::Debug> Tape<T> {
    pub fn dump_nodes(&self) {
        let nodes = self.nodes.borrow();
        let n = nodes.len();
        for (i, node) in nodes.iter().enumerate() {
            println!("[{i}/{n}]: {node:?}");
        }
    }
}

impl<'a, T: Tensor> std::ops::Add for TapeTerm<'a, T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(feature = "expr_name")]
        let name = {
            let nodes = self.tape.nodes.borrow();
            format!(
                "({} + {})",
                nodes[self.idx as usize].name, nodes[rhs.idx as usize].name
            )
        };
        #[cfg(not(feature = "expr_name"))]
        let name = {
            let nodes = self.tape.nodes.borrow();
            nodes[self.idx as usize].name.clone()
        };
        self.tape.term_name(name, TapeValue::Add(self.idx, rhs.idx))
    }
}

impl<'a, T: Tensor> std::ops::Sub for TapeTerm<'a, T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        #[cfg(feature = "expr_name")]
        let name = {
            let nodes = self.tape.nodes.borrow();
            format!(
                "({} - {})",
                nodes[self.idx as usize].name, nodes[rhs.idx as usize].name
            )
        };
        #[cfg(not(feature = "expr_name"))]
        let name = {
            let nodes = self.tape.nodes.borrow();
            nodes[self.idx as usize].name.clone()
        };
        self.tape.term_name(name, TapeValue::Sub(self.idx, rhs.idx))
    }
}

impl<'a, T: Tensor> std::ops::Mul for TapeTerm<'a, T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(feature = "expr_name")]
        let name = {
            let nodes = self.tape.nodes.borrow();
            format!(
                "{} * {}",
                nodes[self.idx as usize].name, nodes[rhs.idx as usize].name
            )
        };
        #[cfg(not(feature = "expr_name"))]
        let name = {
            let nodes = self.tape.nodes.borrow();
            nodes[self.idx as usize].name.clone()
        };
        self.tape.term_name(name, TapeValue::Mul(self.idx, rhs.idx))
    }
}

impl<'a, T: Tensor> std::ops::Div for TapeTerm<'a, T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        #[cfg(feature = "expr_name")]
        let name = {
            let nodes = self.tape.nodes.borrow();
            format!(
                "{} / {}",
                nodes[self.idx as usize].name, nodes[rhs.idx as usize].name
            )
        };
        #[cfg(not(feature = "expr_name"))]
        let name = {
            let nodes = self.tape.nodes.borrow();
            nodes[self.idx as usize].name.clone()
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
    pub fn name(&self) -> String {
        let nodes = self.tape.nodes.borrow();
        nodes[self.idx as usize].name.clone()
    }

    pub fn eval(&self) -> T {
        let mut nodes = self.tape.nodes.borrow_mut();
        clear(&mut nodes);
        let callback: Option<&fn(&[TapeNode<T>], TapeIndex)> = None;
        eval(&mut nodes, self.idx, callback)
    }

    pub fn eval_cb(&self, callback: &impl Fn(&[TapeNode<T>], TapeIndex)) -> T {
        let mut nodes = self.tape.nodes.borrow_mut();
        clear(&mut nodes);
        clear_grad(&mut nodes);
        eval(&mut nodes, self.idx, Some(callback))
    }

    pub fn eval_noclear(&self) -> T {
        let mut nodes = self.tape.nodes.borrow_mut();
        let callback: Option<&fn(&[TapeNode<T>], TapeIndex)> = None;
        eval(&mut nodes, self.idx, callback)
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

    pub fn gen_graph(&self, var: &Self) -> Option<Self> {
        let new_node = {
            let mut nodes = self.tape.nodes.borrow_mut();
            gen_graph(&mut nodes, self.idx, var.idx, &|_, _, _| (), false)
        };
        new_node.map(|idx| TapeTerm {
            tape: self.tape,
            idx,
        })
    }

    pub fn gen_graph_optim(&self, var: &Self, optim: bool) -> Option<Self> {
        let new_node = {
            let mut nodes = self.tape.nodes.borrow_mut();
            gen_graph(&mut nodes, self.idx, var.idx, &|_, _, _| (), optim)
        };
        new_node.map(|idx| TapeTerm {
            tape: self.tape,
            idx,
        })
    }

    pub fn gen_graph_cb(
        &self,
        var: &Self,
        cb: &impl Fn(&[TapeNode<T>], TapeIndex, TapeIndex),
        optim: bool,
    ) -> Option<Self> {
        let new_node = {
            let mut nodes = self.tape.nodes.borrow_mut();
            gen_graph(&mut nodes, self.idx, var.idx, cb, optim)
        };
        new_node.map(|idx| TapeTerm {
            tape: self.tape,
            idx,
        })
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
                f: Some(Box::new(PtrUnaryFn {
                    name: self_name,
                    f,
                    grad,
                })),
            }),
        )
    }

    /// Apply a function, its derivative and a transposition
    pub fn apply_t(&self, f: Box<dyn UnaryFn<T>>) -> Self {
        let self_name = self.tape.nodes.borrow()[self.idx as usize].name.clone();
        let name = format!("{}({})", f.name(), self_name);
        self.tape.term_name(
            name,
            TapeValue::UnaryFn(UnaryFnPayload {
                term: self.idx,
                f: Some(f),
            }),
        )
    }

    pub fn apply_bin(&self, rhs: TapeTerm, f: Box<dyn BinaryFn<T>>) -> Self {
        let self_name = self.tape.nodes.borrow()[self.idx as usize].name.clone();
        #[cfg(feature = "expr_name")]
        let name = format!("{}({})", f.name(), self_name);
        #[cfg(not(feature = "expr_name"))]
        let name = self_name;
        self.tape.term_name(
            name,
            TapeValue::BinaryFn(BinaryFnPayload {
                lhs: self.idx,
                rhs: rhs.idx,
                f: Some(f),
            }),
        )
    }

    pub fn set(&self, value: T) -> Result<(), RustogradError> {
        let mut nodes = self.tape.nodes.borrow_mut();
        let node = nodes.get_mut(self.idx as usize).ok_or_else(|| RangeError)?;
        match &mut node.value {
            TapeValue::Value(val) => *val = value,
            _ => return Err(TermWasNotValueError.into()),
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
        callback: &impl Fn(&[TapeNode<T>], TapeIndex),
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
            connect_to: None,
            output_node: vec![],
            precision: 2,
        }
    }

    pub fn data(&self) -> Option<T> {
        self.tape.nodes.borrow()[self.idx as usize].data.clone()
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
    idx: TapeIndex,
    callback: Option<&impl Fn(&[TapeNode<T>], TapeIndex)>,
) -> T {
    use TapeValue::*;
    if let Some(ref data) = nodes[idx as usize].data {
        return data.clone();
    }
    let data = match nodes[idx as usize].value {
        Value(ref val) => val.clone(),
        Add(lhs, rhs) => eval(nodes, lhs, callback) + eval(nodes, rhs, callback),
        Sub(lhs, rhs) => eval(nodes, lhs, callback) - eval(nodes, rhs, callback),
        Mul(lhs, rhs) => eval(nodes, lhs, callback) * eval(nodes, rhs, callback),
        Div(lhs, rhs) => eval(nodes, lhs, callback) / eval(nodes, rhs, callback),
        Neg(term) => -eval(nodes, term, callback),
        UnaryFn(UnaryFnPayload { term, .. }) => {
            let val = eval(nodes, term, callback);
            // Ugly re-matching to avoid borrow checker
            let UnaryFn(UnaryFnPayload { f, .. }) = &nodes[idx as usize].value else {
                unreachable!()
            };
            f.as_ref().unwrap().f(val)
        }
        BinaryFn(BinaryFnPayload { lhs, rhs, .. }) => {
            let vlhs = eval(nodes, lhs, callback);
            let vrhs = eval(nodes, rhs, callback);
            // Ugly re-matching to avoid borrow checker
            let BinaryFn(BinaryFnPayload { f, .. }) = &nodes[idx as usize].value else {
                unreachable!()
            };
            f.as_ref().unwrap().f(vlhs, vrhs)
        }
    };
    nodes[idx as usize].data = Some(data.clone());
    if let Some(callback) = callback {
        callback(nodes, idx);
    }
    data
}

fn value<T: Clone>(nodes: &[TapeNode<T>], idx: TapeIndex) -> Option<T> {
    nodes[idx as usize].data.clone()
}

/// wrt - The variable to derive With Respect To
fn derive<T: Tensor>(nodes: &mut [TapeNode<T>], idx: TapeIndex, wrt: TapeIndex) -> Option<T> {
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
        UnaryFn(UnaryFnPayload { term, ref f }) => {
            f.as_ref().unwrap().grad(value(nodes, term)?) * derive(nodes, term, wrt)?
        }
        BinaryFn(BinaryFnPayload { lhs, rhs, ref f }) => {
            let local_grad = f
                .as_ref()
                .unwrap()
                .grad(value(nodes, lhs)?, value(nodes, rhs)?);
            let dlhs = derive(nodes, lhs, wrt);
            let drhs = derive(nodes, rhs, wrt);
            match (dlhs, drhs) {
                (Some(dlhs), Some(drhs)) => local_grad.1 * drhs + local_grad.0 * dlhs,
                (None, Some(drhs)) => local_grad.1 * drhs,
                (Some(dlhs), None) => local_grad.0 * dlhs,
                _ => return None,
            }
        }
    };
    Some(grad)
}

fn add_node<T: Tensor>(
    nodes: &mut Vec<TapeNode<T>>,
    name: String,
    value: TapeValue<T>,
) -> TapeIndex {
    let new_idx = nodes.len();
    nodes.push(TapeNode {
        name,
        value,
        data: None,
        grad: None,
    });
    new_idx as TapeIndex
}

fn find_node<T: Tensor>(
    nodes: &[TapeNode<T>],
    pred: impl Fn(&TapeValue<T>) -> bool,
) -> Option<TapeIndex> {
    nodes
        .iter()
        .enumerate()
        .find(|(_, node)| pred(&node.value))
        .map(|(i, _)| i as u32)
}

pub fn add_value<T: Tensor>(nodes: &mut Vec<TapeNode<T>>, val: T) -> TapeIndex {
    let name = format!("{val}");
    add_node(nodes, name, TapeValue::Value(val))
}

pub fn add_add<T: Tensor>(
    nodes: &mut Vec<TapeNode<T>>,
    lhs: TapeIndex,
    rhs: TapeIndex,
    optim: bool,
) -> TapeIndex {
    if optim {
        if let Some(idx) = find_node(nodes, |existing| {
            if let TapeValue::Add(elhs, erhs) = *existing {
                elhs == lhs && erhs == rhs
            } else {
                false
            }
        }) {
            return idx;
        }
    }
    let name = format!(
        "({} + {})",
        nodes[lhs as usize].name, nodes[rhs as usize].name
    );
    add_node(nodes, name, TapeValue::Add(lhs, rhs))
}

pub fn add_sub<T: Tensor>(
    nodes: &mut Vec<TapeNode<T>>,
    lhs: TapeIndex,
    rhs: TapeIndex,
    optim: bool,
) -> TapeIndex {
    if optim {
        if let Some(idx) = find_node(nodes, |existing| {
            if let TapeValue::Sub(elhs, erhs) = *existing {
                elhs == lhs && erhs == rhs
            } else {
                false
            }
        }) {
            return idx;
        }
    }
    let name = format!(
        "({} - {})",
        nodes[lhs as usize].name, nodes[rhs as usize].name
    );
    add_node(nodes, name, TapeValue::Sub(lhs, rhs))
}

pub fn add_mul<T: Tensor>(
    nodes: &mut Vec<TapeNode<T>>,
    lhs: TapeIndex,
    rhs: TapeIndex,
    optim: bool,
) -> TapeIndex {
    if optim {
        if let Some(idx) = find_node(nodes, |existing| {
            if let TapeValue::Mul(elhs, erhs) = *existing {
                elhs == lhs && erhs == rhs
            } else {
                false
            }
        }) {
            return idx;
        }
    }
    let name = format!(
        "{} * {}",
        nodes[lhs as usize].name, nodes[rhs as usize].name
    );
    add_node(nodes, name, TapeValue::Mul(lhs, rhs))
}

pub fn add_div<T: Tensor>(
    nodes: &mut Vec<TapeNode<T>>,
    lhs: TapeIndex,
    rhs: TapeIndex,
    optim: bool,
) -> TapeIndex {
    if optim {
        if let Some(idx) = find_node(nodes, |existing| {
            if let TapeValue::Div(elhs, erhs) = *existing {
                elhs == lhs && erhs == rhs
            } else {
                false
            }
        }) {
            return idx;
        }
    }
    let name = format!(
        "{} / {}",
        nodes[lhs as usize].name, nodes[rhs as usize].name
    );
    add_node(nodes, name, TapeValue::Div(lhs, rhs))
}

pub fn add_neg<T: Tensor>(nodes: &mut Vec<TapeNode<T>>, node: TapeIndex, optim: bool) -> TapeIndex {
    if optim {
        if let Some(idx) = find_node(nodes, |existing| {
            if let TapeValue::Neg(e) = *existing {
                e == node
            } else {
                false
            }
        }) {
            return idx;
        }
    }
    let name = format!("-{}", nodes[node as usize].name);
    add_node(nodes, name, TapeValue::Neg(node))
}

pub fn add_unary_fn<T: Tensor>(
    nodes: &mut Vec<TapeNode<T>>,
    f: Box<dyn UnaryFn<T>>,
    node: TapeIndex,
) -> TapeIndex {
    let name = format!("{}({})", f.name(), nodes[node as usize].name);
    add_node(
        nodes,
        name,
        TapeValue::UnaryFn(UnaryFnPayload {
            term: node,
            f: Some(f),
        }),
    )
}

fn gen_graph<T: Tensor + 'static>(
    nodes: &mut Vec<TapeNode<T>>,
    idx: TapeIndex,
    wrt: TapeIndex,
    cb: &impl Fn(&[TapeNode<T>], TapeIndex, TapeIndex),
    optim: bool,
) -> Option<TapeIndex> {
    use TapeValue::*;
    let ret = match nodes[idx as usize].value {
        Value(_) => {
            if idx == wrt {
                Some(1)
            } else {
                None
            }
        }
        Add(lhs, rhs) => {
            let lhs = gen_graph(nodes, lhs, wrt, cb, optim);
            let rhs = gen_graph(nodes, rhs, wrt, cb, optim);
            match (lhs, rhs) {
                (Some(lhs), None) => Some(lhs),
                (None, Some(rhs)) => Some(rhs),
                (Some(lhs), Some(rhs)) => Some(add_add(nodes, lhs, rhs, optim)),
                _ => None,
            }
        }
        Sub(lhs, rhs) => {
            let lhs = gen_graph(nodes, lhs, wrt, cb, optim);
            let rhs = gen_graph(nodes, rhs, wrt, cb, optim);
            match (lhs, rhs) {
                (Some(lhs), None) => Some(lhs),
                (None, Some(rhs)) => Some(add_neg(nodes, rhs, optim)),
                (Some(lhs), Some(rhs)) => Some(add_sub(nodes, lhs, rhs, optim)),
                _ => None,
            }
        }
        Mul(lhs, rhs) => {
            let dlhs = gen_graph(nodes, lhs, wrt, cb, optim);
            let drhs = gen_graph(nodes, rhs, wrt, cb, optim);
            match (dlhs, drhs) {
                (Some(dlhs), None) => Some(add_mul(nodes, dlhs, rhs, optim)),
                (None, Some(drhs)) => Some(add_mul(nodes, lhs, drhs, optim)),
                (Some(dlhs), Some(drhs)) => {
                    let plhs = add_mul(nodes, dlhs, rhs, optim);
                    let prhs = add_mul(nodes, lhs, drhs, optim);
                    let node = add_add(nodes, plhs, prhs, optim);
                    Some(node)
                }
                _ => None,
            }
        }
        Div(lhs, rhs) => {
            let dlhs = gen_graph(nodes, lhs, wrt, cb, optim);
            let drhs = gen_graph(nodes, rhs, wrt, cb, optim);
            match (dlhs, drhs) {
                (Some(dlhs), None) => Some(add_div(nodes, dlhs, rhs, optim)),
                (None, Some(drhs)) => {
                    let node = add_mul(nodes, lhs, drhs, optim);
                    let node = add_div(nodes, node, rhs, optim);
                    let node = add_div(nodes, node, rhs, optim);
                    Some(add_neg(nodes, node, optim))
                }
                (Some(dlhs), Some(drhs)) => {
                    let plhs = add_div(nodes, dlhs, rhs, optim);
                    let node = add_mul(nodes, lhs, drhs, optim);
                    let prhs = add_div(nodes, node, rhs, optim);
                    let prhs = add_div(nodes, prhs, rhs, optim);
                    Some(add_sub(nodes, plhs, prhs, optim))
                }
                _ => None,
            }
        }
        Neg(term) => gen_graph(nodes, term, wrt, cb, optim).map(|node| add_neg(nodes, node, optim)),
        UnaryFn(UnaryFnPayload { term, ref mut f }) => {
            let taken_f = f.take();
            let derived = gen_graph(nodes, term, wrt, cb, optim);
            let ret = derived.and_then(|derived| {
                taken_f
                    .as_ref()
                    .unwrap()
                    .gen_graph(nodes, term, idx, derived, optim)
            });
            if let UnaryFn(UnaryFnPayload { ref mut f, .. }) = nodes[idx as usize].value {
                *f = taken_f;
            } else {
                unreachable!()
            }
            ret
        }
        BinaryFn(BinaryFnPayload {
            lhs,
            rhs,
            ref mut f,
        }) => {
            let taken_f = f.take();
            let dlhs = gen_graph(nodes, lhs, wrt, cb, optim);
            let drhs = gen_graph(nodes, lhs, wrt, cb, optim);
            let ret = match (dlhs, drhs) {
                (Some(dlhs), None) => taken_f
                    .as_ref()
                    .unwrap()
                    .gen_graph(nodes, lhs, 1, idx, dlhs, 0),
                (None, Some(drhs)) => taken_f
                    .as_ref()
                    .unwrap()
                    .gen_graph(nodes, 1, rhs, idx, 0, drhs),
                (Some(dlhs), Some(drhs)) => taken_f
                    .as_ref()
                    .unwrap()
                    .gen_graph(nodes, lhs, rhs, idx, dlhs, drhs),
                _ => None,
            };
            if let BinaryFn(BinaryFnPayload { ref mut f, .. }) = nodes[idx as usize].value {
                *f = taken_f;
            } else {
                unreachable!()
            }
            ret
        }
    };
    if let Some(generated) = ret {
        cb(nodes, idx, generated);
    }
    ret
}

fn clear_grad<T: Tensor>(nodes: &mut [TapeNode<T>]) {
    for node in nodes {
        node.grad = None;
    }
}

fn backprop_set<T: Tensor>(
    nodes: &mut [TapeNode<T>],
    idx: TapeIndex,
    grad: T,
    callback: &impl Fn(&[TapeNode<T>], TapeIndex),
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
    idx: TapeIndex,
    callback: &impl Fn(&[TapeNode<T>], TapeIndex),
) -> Result<(), ValueNotDefinedError> {
    use TapeValue::*;
    nodes[idx as usize].grad = Some(T::one());
    callback(nodes, idx);
    for i in (0..=idx).rev() {
        let Some(grad) = nodes[i as usize].grad.as_ref().map(Clone::clone) else {
            continue;
        };
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
            UnaryFn(UnaryFnPayload { term, ref f }) => {
                let val = value(nodes, term).ok_or(ValueNotDefinedError)?;
                let f = f.as_ref().unwrap();
                let newgrad = grad * f.grad(val);
                let endgrad = f.t(newgrad);
                backprop_set(nodes, term, endgrad, callback)
            }
            BinaryFn(BinaryFnPayload { lhs, rhs, ref f }) => {
                let vlhs = value(nodes, lhs).ok_or(ValueNotDefinedError)?;
                let vrhs = value(nodes, rhs).ok_or(ValueNotDefinedError)?;
                let f = f.as_ref().unwrap();
                let local_grad = f.grad(vlhs, vrhs);
                let (grad_l, grad_r) = f.t(grad);
                let newgrad_l = local_grad.0 * grad_l;
                let newgrad_r = local_grad.1 * grad_r;
                backprop_set(nodes, lhs, newgrad_l, callback);
                backprop_set(nodes, rhs, newgrad_r, callback);
            }
        }
    }
    Ok(())
}

/// The dot file writer configuration builder with the builder pattern.
pub struct TapeDotBuilder<'a, T: Default> {
    this: TapeTerm<'a, T>,
    show_values: bool,
    vertical: bool,
    hilight: Option<TapeIndex>,
    connect_to: Option<TapeIndex>,
    output_node: Vec<(TapeIndex, String)>,
    precision: usize,
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
    pub fn highlights(mut self, term: TapeIndex) -> Self {
        self.hilight = Some(term);
        self
    }

    /// Specify the node that has connection from highlighted node.
    pub fn connect_to(mut self, term: TapeIndex) -> Self {
        self.connect_to = Some(term);
        self
    }

    /// Add an output node to indicate in the graph. Useful to distinguish the output in a complex graph.
    pub fn output_node(mut self, term: TapeIndex, name: impl Into<String>) -> Self {
        self.output_node.push((term, name.into()));
        self
    }

    /// Add an output node to indicate in the graph. Useful to distinguish the output in a complex graph.
    pub fn output_term(mut self, term: TapeTerm<'a>, name: impl Into<String>) -> Self {
        self.output_node.push((term.idx, name.into()));
        self
    }

    /// Set floating point precision after decimal point
    pub fn precision(mut self, precision: usize) -> Self {
        self.precision = precision;
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
            "digraph G {{\nrankdir=\"{}\";
            newrank=true;",
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
            let border = if self.hilight.is_some_and(|x| x == id as TapeIndex) {
                " color=red penwidth=2"
            } else {
                ""
            };
            let label = if self.show_values {
                let formatter = |v| format!("{v:.precision$}", precision = self.precision);
                format!(
                    "\\ndata:{}, grad:{}",
                    term.data
                        .as_ref()
                        .map(formatter)
                        .unwrap_or_else(|| "None".into()),
                    term.grad
                        .as_ref()
                        .map(formatter)
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
                Add(lhs, rhs)
                | Sub(lhs, rhs)
                | Mul(lhs, rhs)
                | Div(lhs, rhs)
                | BinaryFn(BinaryFnPayload { lhs, rhs, .. }) => [Some(lhs), Some(rhs)],
                Neg(term) | UnaryFn(UnaryFnPayload { term, .. }) => [Some(term), None],
            };
            for pid in parents.into_iter().filter_map(|v| v) {
                writeln!(writer, "a{} -> a{};", pid, id)?;
            }
        }
        if let Some((from, to)) = self.hilight.zip(self.connect_to) {
            writeln!(
                writer,
                "a{} -> a{} [ style=\"dashed,bold\" color=green constraint=false ];",
                from, to
            )?;
        }
        for (i, (output, name)) in self.output_node.iter().enumerate() {
            writeln!(writer, "output{i} [label=\"{name}\" shape=oval];")?;
            writeln!(writer, "a{output} -> output{i} [ style=\"bold\" ];",)?;
        }
        writeln!(writer, "}}")?;
        Ok(())
    }
}
