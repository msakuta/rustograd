//! gen_graph example with polynomials.
use std::io::Write;

use rustograd::{
    tape::{add_mul, add_unary_fn, add_value, TapeIndex, TapeNode},
    Tape, UnaryFn,
};

struct PolynomialFn(i32);

impl UnaryFn<f64> for PolynomialFn {
    fn name(&self) -> String {
        format!("(^{})", self.0)
    }

    fn f(&self, data: f64) -> f64 {
        data.powi(self.0)
    }

    fn grad(&self, data: f64) -> f64 {
        self.0 as f64 * data.powi(self.0 - 1)
    }

    fn gen_graph(
        &self,
        nodes: &mut Vec<TapeNode<f64>>,
        input: TapeIndex,
        _output: TapeIndex,
        _derived: TapeIndex,
        optim: bool,
    ) -> Option<TapeIndex> {
        if self.0 == 0 {
            None
        } else if 1 < self.0 {
            let poly = if self.0 == 2 {
                input
            } else {
                add_unary_fn(nodes, Box::new(Self(self.0 - 1)), input)
            };
            let factor = add_value(nodes, self.0 as f64);
            Some(add_mul(nodes, factor, poly, optim))
        } else {
            Some(1)
        }
    }
}

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 1.23);
    let sin_a = (a).apply_t(Box::new(PolynomialFn(3)));

    let mut csv = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    let mut derivatives = vec![sin_a];
    let mut next = sin_a;
    write!(csv, "x, $x^3$, ").unwrap();
    for i in 1..4 {
        let Some(next_i) = next.gen_graph(&a) else {
            break;
        };
        next = next_i;
        derivatives.push(next);
        write!(csv, "$(d^{i} x^3)/(d x^{i})$, ").unwrap();
    }
    writeln!(csv, "").unwrap();

    for i in -50..50 {
        let x = i as f64 * 0.02;
        a.set(x).unwrap();
        write!(csv, "{x}, ").unwrap();
        for d in &derivatives {
            write!(csv, "{}, ", d.eval()).unwrap();
        }
        writeln!(csv, "").unwrap();
    }

    let mut writer = std::io::BufWriter::new(std::fs::File::create("graph.dot").unwrap());
    derivatives
        .last()
        .unwrap()
        .dot_builder()
        .show_values(true)
        .dot(&mut writer)
        .unwrap();
}
