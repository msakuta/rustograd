use rustograd::Tape;

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 2.);
    let b = tape.term("b", 1.);
    // let c = tape.term("b", 42.);
    let aba = (a - b) * a;
    let daba = aba.gen_graph(&a).unwrap();
    let dabab = daba + b;
    aba.eval();
    aba.backprop().unwrap();
    dabab.eval_noclear();

    dabab
        .dot_builder()
        .show_values(true)
        .dot(&mut std::io::stdout())
        .unwrap();
}
