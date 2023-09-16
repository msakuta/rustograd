//! gen_graph example with polynomials.
use std::io::Write;

use rustograd::{
    tape::{add_add, add_div, add_mul, add_neg, TapeIndex, TapeNode},
    BinaryFn, Tape,
};

struct Atan2Fn;

impl BinaryFn<f64> for Atan2Fn {
    fn name(&self) -> String {
        format!("atan2")
    }

    fn f(&self, lhs: f64, rhs: f64) -> f64 {
        lhs.atan2(rhs)
    }

    fn grad(&self, lhs: f64, rhs: f64) -> (f64, f64) {
        let denom = lhs.powi(2) + rhs.powi(2);
        (rhs / denom, -lhs / denom)
    }

    fn t(&self, data: f64) -> (f64, f64) {
        (data, data)
    }

    fn gen_graph(
        &self,
        nodes: &mut Vec<TapeNode<f64>>,
        lhs: TapeIndex,
        rhs: TapeIndex,
        _output: TapeIndex,
        _l_derived: TapeIndex,
        _r_derived: TapeIndex,
    ) -> Option<TapeIndex> {
        let lhs2 = add_mul(nodes, lhs, lhs, true);
        let rhs2 = add_mul(nodes, rhs, rhs, true);
        let denom = add_add(nodes, lhs2, rhs2, true);
        let dlhs = add_div(nodes, rhs, denom, true);
        let drhs = add_div(nodes, lhs, denom, true);
        let drhs = add_neg(nodes, drhs, true);
        Some(add_add(nodes, dlhs, drhs, true))
    }
}

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 1.23);
    let b = tape.term("b", 0.93);
    let atan2_ab = (a).apply_bin(b, Box::new(Atan2Fn));

    let mut csv = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    let mut derivatives = vec![atan2_ab];
    let mut next = atan2_ab;
    write!(csv, "\"theta\", \"atan2(y x)\", \"d (atan2(y x)) / d y (derive)\", \"d (atan2(y x)) / d x (derive)\", \"d (atan2(y x)) / d y (backprop)\", \"d (atan2(y x)) / d x (backprop)\", ").unwrap();
    for i in 1..4 {
        let Some(next_i) = next.gen_graph(&a) else {
            break;
        };
        next = next_i;
        derivatives.push(next);
        write!(csv, "$(d^{i} x^3)/(d x^{i})$, ").unwrap();
    }
    writeln!(csv, "").unwrap();

    for i in 0..100 {
        let angle = i as f64 * std::f64::consts::PI * 2. / 100.;
        a.set(angle.sin()).unwrap();
        b.set(angle.cos()).unwrap();
        write!(csv, "{angle}, ").unwrap();
        let eval = atan2_ab.eval();
        write!(csv, "{eval}, ").unwrap();
        let grad_a = atan2_ab.derive(&a).unwrap();
        write!(csv, "{grad_a}, ").unwrap();
        let grad_b = atan2_ab.derive(&b).unwrap();
        write!(csv, "{grad_b}, ").unwrap();
        atan2_ab.backprop().unwrap();
        write!(csv, "{}, {}, ", a.grad().unwrap(), b.grad().unwrap()).unwrap();
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
