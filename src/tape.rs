//! Implementation of shared memory arena for the terms, aka a tape.
//! See https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

use std::{cell::RefCell, io::Write};

#[derive(Default, Debug)]
pub struct Tape {
    nodes: RefCell<Vec<TapeNode>>,
}

#[derive(Clone, Debug)]
struct TapeNode {
    name: String,
    value: TapeValue,
    data: f64,
    grad: f64,
}

#[derive(Clone, Debug)]
struct UnaryFnPayload {
    term: u32,
    f: fn(f64) -> f64,
    grad: fn(f64) -> f64,
}

#[derive(Clone, Debug)]
enum TapeValue {
    Value(f64),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Div(u32, u32),
    Neg(u32),
    UnaryFn(UnaryFnPayload),
}

#[derive(Copy, Clone)]
pub struct TapeTerm<'a> {
    tape: &'a Tape,
    idx: u32,
}

impl Tape {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn term<'a>(&'a self, name: impl Into<String>, init: f64) -> TapeTerm<'a> {
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(TapeNode {
            name: name.into(),
            value: TapeValue::Value(init),
            data: 0.,
            grad: 0.,
        });
        TapeTerm {
            tape: self,
            idx: idx as u32,
        }
    }

    fn term0<'a>(&'a self, value: TapeValue) -> TapeTerm<'a> {
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(TapeNode {
            name: format!("a{idx}"),
            value,
            data: 0.,
            grad: 0.,
        });
        TapeTerm {
            tape: self,
            idx: idx as u32,
        }
    }

    fn term_name<'a>(&'a self, name: impl Into<String>, value: TapeValue) -> TapeTerm<'a> {
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(TapeNode {
            name: name.into(),
            value,
            data: 0.,
            grad: 0.,
        });
        TapeTerm {
            tape: self,
            idx: idx as u32,
        }
    }
}

impl<'a> std::ops::Add for TapeTerm<'a> {
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

impl<'a> std::ops::Sub for TapeTerm<'a> {
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

impl<'a> std::ops::Mul for TapeTerm<'a> {
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

impl<'a> std::ops::Div for TapeTerm<'a> {
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

impl<'a> std::ops::Neg for TapeTerm<'a> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let name = {
            let nodes = self.tape.nodes.borrow();
            format!("-{}", nodes[self.idx as usize].name)
        };
        self.tape.term_name(name, TapeValue::Neg(self.idx))
    }
}

impl<'a> TapeTerm<'a> {
    pub fn eval(&self) -> f64 {
        let mut nodes = self.tape.nodes.borrow_mut();
        eval(&mut nodes, self.idx)
    }

    /// One-time derivation. Does not update internal gradient values.
    pub fn derive(&self, var: &Self) -> f64 {
        if self.idx == var.idx {
            1.
        } else {
            let mut nodes = self.tape.nodes.borrow_mut();
            derive(&mut nodes, self.idx, var.idx)
        }
    }

    pub fn apply(
        &self,
        name: &(impl AsRef<str> + ?Sized),
        f: fn(f64) -> f64,
        grad: fn(f64) -> f64,
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

    pub fn set(&self, value: f64) -> Result<(), ()> {
        let mut nodes = self.tape.nodes.borrow_mut();
        let node = nodes.get_mut(self.idx as usize).ok_or_else(|| ())?;
        match &mut node.value {
            TapeValue::Value(val) => *val = value,
            _ => return Err(()),
        }
        Ok(())
    }

    pub fn backprop(&self) {
        let mut nodes = self.tape.nodes.borrow_mut();
        clear_grad(&mut nodes);
        backprop_rec(&mut nodes, self.idx, 1.);
    }

    /// Write graphviz dot file to the given writer.
    pub fn dot(&self, writer: &mut impl Write) -> std::io::Result<()> {
        let nodes = self.tape.nodes.borrow();
        writeln!(writer, "digraph G {{\nrankdir=\"LR\";")?;
        for (id, term) in nodes.iter().enumerate() {
            let color = if term.grad != 0. {
                "style=filled fillcolor=\"#ffff7f\""
            } else if term.data != 0. {
                "style=filled fillcolor=\"#7fff7f\""
            } else {
                ""
            };
            writeln!(
                writer,
                "a{} [label=\"{} \\ndata:{}, grad:{}\" shape=rect {color}];",
                id, term.name, term.data, term.grad
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

    pub fn grad(&self) -> f64 {
        self.tape.nodes.borrow()[self.idx as usize].grad
    }
}

fn eval(nodes: &mut [TapeNode], idx: u32) -> f64 {
    use TapeValue::*;
    let data = match nodes[idx as usize].value {
        Value(val) => val,
        Add(lhs, rhs) => eval(nodes, lhs) + eval(nodes, rhs),
        Sub(lhs, rhs) => eval(nodes, lhs) - eval(nodes, rhs),
        Mul(lhs, rhs) => eval(nodes, lhs) * eval(nodes, rhs),
        Div(lhs, rhs) => eval(nodes, lhs) / eval(nodes, rhs),
        Neg(term) => -eval(nodes, term),
        UnaryFn(UnaryFnPayload { term, f, .. }) => f(eval(nodes, term)),
    };
    nodes[idx as usize].data = data;
    data
}

fn value(nodes: &[TapeNode], idx: u32) -> f64 {
    nodes[idx as usize].data
}

/// wrt - The variable to derive With Respect To
fn derive(nodes: &mut [TapeNode], idx: u32, wrt: u32) -> f64 {
    use TapeValue::*;
    // println!("derive({}, {}): {:?}", idx, wrt, nodes[idx as usize].value);
    let grad = match nodes[idx as usize].value {
        Value(_) => {
            if idx == wrt {
                1.
            } else {
                0.
            }
        }
        Add(lhs, rhs) => derive(nodes, lhs, wrt) + derive(nodes, rhs, wrt),
        Sub(lhs, rhs) => derive(nodes, lhs, wrt) - derive(nodes, rhs, wrt),
        Mul(lhs, rhs) => {
            let dlhs = derive(nodes, lhs, wrt);
            let drhs = derive(nodes, rhs, wrt);
            dlhs * value(nodes, rhs) + value(nodes, lhs) * drhs
        }
        Div(lhs, rhs) => {
            let dlhs = derive(nodes, lhs, wrt);
            let drhs = derive(nodes, rhs, wrt);
            let elhs = value(nodes, lhs);
            let erhs = value(nodes, rhs);
            dlhs / erhs - elhs / erhs / erhs * drhs
        }
        Neg(term) => -derive(nodes, term, wrt),
        UnaryFn(UnaryFnPayload { term, grad, .. }) => {
            grad(value(nodes, term)) * derive(nodes, term, wrt)
        }
    };
    grad
}

fn clear_grad(nodes: &mut [TapeNode]) {
    for node in nodes {
        node.grad = 0.;
    }
}

/// Assign gradient to all nodes
fn backprop_rec(nodes: &mut [TapeNode], idx: u32, grad: f64) {
    use TapeValue::*;
    nodes[idx as usize].grad += grad;
    match nodes[idx as usize].value {
        Value(_) => (),
        Add(lhs, rhs) => {
            backprop_rec(nodes, lhs, grad);
            backprop_rec(nodes, rhs, grad);
        }
        Sub(lhs, rhs) => {
            backprop_rec(nodes, lhs, grad);
            backprop_rec(nodes, rhs, -grad);
        }
        Mul(lhs, rhs) => {
            let erhs = value(nodes, rhs);
            let elhs = value(nodes, lhs);
            backprop_rec(nodes, lhs, grad * erhs);
            backprop_rec(nodes, rhs, grad * elhs);
        }
        Div(lhs, rhs) => {
            let erhs = value(nodes, rhs);
            let elhs = value(nodes, lhs);
            backprop_rec(nodes, lhs, grad / erhs);
            backprop_rec(nodes, rhs, -grad * elhs / erhs / erhs);
        }
        Neg(term) => backprop_rec(nodes, term, -grad),
        UnaryFn(UnaryFnPayload { term, grad: g, .. }) => {
            let val = value(nodes, term);
            backprop_rec(nodes, term, grad * g(val))
        }
    }
}
