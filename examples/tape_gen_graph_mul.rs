use rustograd::Tape;

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 2.);
    let b = tape.term("b", 1.);
    // let c = tape.term("b", 42.);
    let aba = (a - b) * a;
    let daba = aba.gen_graph(&a).unwrap();
    aba.eval();
    aba.backprop().unwrap();
    daba.eval_noclear();

    let ddaba = daba.gen_graph(&a).unwrap();
    ddaba.eval_noclear();

    daba.dot_builder()
        .show_values(true)
        .dot(&mut std::io::stdout())
        .unwrap();
}
