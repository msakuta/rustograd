use rustograd::{tape::TapeIndex, Tape};

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 2.);
    let b = tape.term("b", 1.);
    // let c = tape.term("b", 42.);
    let aba = (a - b) * a;

    let counter = std::cell::Cell::new(0);
    let next_file_writer = || {
        let i = counter.get();
        let file = std::io::BufWriter::new(std::fs::File::create(format!("dot{i}.dot")).unwrap());
        counter.set(i + 1);
        file
    };
    let callback = |output2: Option<TapeIndex>| {
        move |nodes: &_, idx, generated| {
            let mut file = next_file_writer();
            let mut dot_builder = aba
                .dot_builder()
                .vertical(true)
                .show_values(false)
                .highlights(idx)
                .connect_to(generated)
                .output_node(aba.to_tape_index(), "f(a)");
            if let Some(output2) = output2 {
                dot_builder = dot_builder.output_node(output2, "f'(a)")
            }
            dot_builder.dot_borrowed(nodes, &mut file).unwrap();
        }
    };

    let daba = aba.gen_graph_cb(&a, &callback(None), false).unwrap();
    aba.eval();
    aba.backprop().unwrap();
    daba.eval_noclear();

    let ddaba = daba
        .gen_graph_cb(&a, &callback(Some(daba.to_tape_index())), false)
        .unwrap();
    ddaba.eval_noclear();

    daba.dot_builder()
        .show_values(true)
        .vertical(true)
        .show_values(false)
        .output_node(aba.to_tape_index(), "f(a)")
        .output_node(daba.to_tape_index(), "f'(a)")
        .output_node(ddaba.to_tape_index(), "f''(a)")
        .dot(&mut next_file_writer())
        .unwrap();
}
