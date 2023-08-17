use std::io::Write;

use rustograd::{
    tape::{add_mul, TapeNode},
    Tape, UnaryFn,
};

struct ExpFn;

impl UnaryFn<f64> for ExpFn {
    fn name(&self) -> String {
        "exp".to_string()
    }

    fn f(&self, data: f64) -> f64 {
        data.exp()
    }

    fn grad(&self, data: f64) -> f64 {
        data.exp()
    }

    fn gen_graph(
        &self,
        nodes: &mut Vec<TapeNode<f64>>,
        idx: u32,
        _input: u32,
        derived: u32,
    ) -> Option<u32> {
        Some(add_mul(nodes, idx, derived))
    }
}

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 1.23);
    let exp_a = (-a * a).apply_t(Box::new(ExpFn));

    let mut csv = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    let mut derivatives = vec![exp_a];
    let mut next = exp_a;
    write!(csv, "x, $e^x$, ").unwrap();
    for i in 1..4 {
        next = next.gen_graph(&a).unwrap();
        derivatives.push(next);
        write!(csv, "$(d^{i} \\exp(-x^2)/(d x^{i})$, ").unwrap();
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
        .vertical(true)
        .show_values(false)
        .dot(&mut writer)
        .unwrap();
}
