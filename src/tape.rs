//! Implementation of shared memory arena for the terms, aka a tape.
//! See https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

use std::cell::RefCell;

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
}

impl<'a> std::ops::Add for TapeTerm<'a> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.tape.term0(TapeValue::Add(self.idx, rhs.idx))
    }
}

impl<'a> std::ops::Sub for TapeTerm<'a> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.tape.term0(TapeValue::Sub(self.idx, rhs.idx))
    }
}

impl<'a> std::ops::Mul for TapeTerm<'a> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        self.tape.term0(TapeValue::Mul(self.idx, rhs.idx))
    }
}

impl<'a> std::ops::Div for TapeTerm<'a> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.tape.term0(TapeValue::Div(self.idx, rhs.idx))
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
}

fn eval(nodes: &mut [TapeNode], idx: u32) -> f64 {
    use TapeValue::*;
    let data = match nodes[idx as usize].value {
        Value(val) => val,
        Add(lhs, rhs) => eval(nodes, lhs) + eval(nodes, rhs),
        Sub(lhs, rhs) => eval(nodes, lhs) - eval(nodes, rhs),
        Mul(lhs, rhs) => eval(nodes, lhs) * eval(nodes, rhs),
        Div(lhs, rhs) => eval(nodes, lhs) / eval(nodes, rhs),
        UnaryFn(UnaryFnPayload { term, f, .. }) => f(eval(nodes, term)),
    };
    nodes[idx as usize].data = data;
    data
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
            dlhs * eval(nodes, rhs) + eval(nodes, lhs) * drhs
        }
        Div(lhs, rhs) => {
            let dlhs = derive(nodes, lhs, wrt);
            let drhs = derive(nodes, rhs, wrt);
            if drhs == 0. {
                dlhs / eval(nodes, rhs)
            } else {
                dlhs / eval(nodes, rhs) + eval(nodes, lhs) / drhs
            }
        }
        UnaryFn(UnaryFnPayload { term, grad, .. }) => {
            grad(eval(nodes, term)) * derive(nodes, term, wrt)
        }
    };
    nodes[idx as usize].grad = grad;
    grad
}
