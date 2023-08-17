use std::io::Write;

use rustograd::{
    tape::{add_mul, add_sub, add_value, TapeNode},
    Tape, UnaryFn,
};

struct SigmoidFn;

impl UnaryFn<f64> for SigmoidFn {
    fn name(&self) -> String {
        "exp".to_string()
    }

    fn f(&self, data: f64) -> f64 {
        1. / (1. + (-data).exp())
    }

    fn grad(&self, data: f64) -> f64 {
        let s = 1. / (1. + (-data).exp());
        (1. - s) * s
    }

    fn gen_graph(
        &self,
        nodes: &mut Vec<TapeNode<f64>>,
        idx: u32,
        _input: u32,
        _derived: u32,
    ) -> Option<u32> {
        let one = add_value(nodes, 1.);
        let one_minus_sigmoid = add_sub(nodes, one, idx);
        Some(add_mul(nodes, one_minus_sigmoid, idx))
    }
}

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 1.23);
    let sin_a = (a).apply_t(Box::new(SigmoidFn));

    let mut csv = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    let mut derivatives = vec![sin_a];
    let mut next = sin_a;
    write!(csv, "x, $\\mathrm{{sigmoid}}(x)$, ").unwrap();
    for i in 1..4 {
        next = next.gen_graph(&a).unwrap();
        derivatives.push(next);
        write!(csv, "$(d^{i} \\mathrm{{sigmoid}}(x)/(d x^{i})$, ").unwrap();
    }
    writeln!(csv, "").unwrap();

    for i in -50..50 {
        let x = i as f64 * 0.1;
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