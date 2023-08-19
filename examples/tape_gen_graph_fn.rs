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
    let b = tape.term("b", 3.21);
    let exp_a = (a * b).apply_t(Box::new(ExpFn));
    let d_exp_a = exp_a.gen_graph(&a).unwrap();
    exp_a.eval();
    exp_a.backprop().unwrap();
    d_exp_a.eval_noclear();

    d_exp_a
        .dot_builder()
        .show_values(true)
        .dot(&mut std::io::stdout())
        .unwrap();
}
