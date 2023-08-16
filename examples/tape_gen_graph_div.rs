use rustograd::Tape;

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 1.23);
    let b = tape.term("b", 3.21);
    let abb = (a + b) / b;
    let dabb = abb.gen_graph(&b).unwrap();
    abb.eval();
    abb.backprop().unwrap();
    dabb.eval_noclear();

    dabb.dot_builder()
        .show_values(true)
        .dot(&mut std::io::stdout())
        .unwrap();
}
