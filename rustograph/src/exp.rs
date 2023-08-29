use rustograd::{
    tape::{add_mul, TapeIndex, TapeNode},
    UnaryFn,
};

pub(crate) struct ExpFn;

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
        optim: bool,
    ) -> Option<TapeIndex> {
        Some(add_mul(nodes, output, derived, optim))
    }
}
