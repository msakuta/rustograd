use rustograd::Tape;

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 123.);
    let b = tape.term("b", 321.);
    let aba = a - b + a;
    let daba = aba.gen_graph(&a).unwrap();
    aba.eval();
    aba.backprop().unwrap();
    daba.eval_noclear();

    assert!(daba.gen_graph(&a).is_none());

    daba.dot_builder()
        .show_values(true)
        .dot(&mut std::io::stdout())
        .unwrap();
}
