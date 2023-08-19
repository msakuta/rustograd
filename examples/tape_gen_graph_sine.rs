use std::io::Write;

use rustograd::{
    tape::{add_unary_fn, TapeIndex, TapeNode},
    Tape, UnaryFn,
};

struct SinFn(usize);

impl UnaryFn<f64> for SinFn {
    fn name(&self) -> String {
        match self.0 % 4 {
            0 => "sin",
            1 => "cos",
            2 => "-sin",
            3 => "-cos",
            _ => unreachable!(),
        }
        .to_string()
    }

    fn f(&self, data: f64) -> f64 {
        match self.0 % 4 {
            0 => data.sin(),
            1 => data.cos(),
            2 => -data.sin(),
            3 => -data.cos(),
            _ => unreachable!(),
        }
    }

    fn grad(&self, data: f64) -> f64 {
        Self(self.0 + 1).f(data)
    }

    fn gen_graph(
        &self,
        nodes: &mut Vec<TapeNode<f64>>,
        input: TapeIndex,
        _output: TapeIndex,
        _derived: TapeIndex,
    ) -> Option<TapeIndex> {
        Some(add_unary_fn(nodes, Box::new(Self(self.0 + 1)), input))
    }
}

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 1.23);
    let sin_a = (a).apply_t(Box::new(SinFn(0)));

    let mut csv = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    let mut derivatives = vec![sin_a];
    let mut next = sin_a;
    write!(csv, "x, $\\sin(x)$, ").unwrap();
    for i in 1..4 {
        next = next.gen_graph(&a).unwrap();
        derivatives.push(next);
        write!(csv, "$(d^{i} \\sin(x)/(d x^{i})$, ").unwrap();
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
