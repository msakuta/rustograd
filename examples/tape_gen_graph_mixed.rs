use rustograd::Tape;

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 2.);
    let b = tape.term("b", 1.);
    // let c = tape.term("b", 42.);
    let aba = (a - b) * a;
    let daba = aba.gen_graph(&a).unwrap();
    let dabab = daba + aba;
    aba.eval();
    aba.backprop().unwrap();
    dabab.eval_noclear();

    dabab
        .dot_builder()
        .vertical(true)
        .show_values(false)
        .output_node(aba.to_tape_index(), "g(a)")
        .output_node(daba.to_tape_index(), "g'(a)")
        .output_node(dabab.to_tape_index(), "f'(a)")
        .dot(&mut std::io::stdout())
        .unwrap();
}
