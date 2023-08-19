use std::io::Write;

use rustograd::{
    tape::{add_mul, TapeIndex, TapeNode},
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
        _input: TapeIndex,
        output: TapeIndex,
        derived: TapeIndex,
    ) -> Option<TapeIndex> {
        Some(add_mul(nodes, output, derived))
    }
}

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 1.23);
    let exp_a = (-a * a).apply_t(Box::new(ExpFn));

    let next = std::cell::Cell::new(exp_a);

    let counter = std::cell::Cell::new(0);
    let callback = |nodes: &_, idx, generated| {
        let i = counter.get();
        let mut file =
            std::io::BufWriter::new(std::fs::File::create(format!("dot{i}.dot")).unwrap());
        next.get()
            .dot_builder()
            .vertical(true)
            .show_values(false)
            .highlights(idx)
            .connect_to(generated)
            .dot_borrowed(nodes, &mut file)
            .unwrap();
        counter.set(i + 1);
    };

    let mut csv = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    let mut derivatives = vec![exp_a];
    write!(csv, "x, $e^x$, ").unwrap();
    for i in 1..4 {
        let next_node = next.get().gen_graph_cb(&a, &callback).unwrap();
        derivatives.push(next_node);
        write!(csv, "$(d^{i} \\exp(-x^2)/(d x^{i})$, ").unwrap();
        next.set(next_node);
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
