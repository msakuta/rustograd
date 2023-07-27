use rustograd::{Tape, TapeTerm};

use std::io::Write;

fn main() {
    let tape = Tape::new();
    let (x, all) = build_model(&tape);
    let mut file = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    for i in -40..=40 {
        let xval = i as f64 / 20. * std::f64::consts::PI;
        x.set(xval).unwrap();
        writeln!(file, "{xval}, {}, {}", all.eval(), all.derive(&x)).unwrap();
    }
    x.set(0.).unwrap();
    all.backprop();
    println!("tape: {tape:?}");
    all.dot(&mut std::io::stdout()).unwrap();
}

fn build_model(tape: &Tape) -> (TapeTerm, TapeTerm) {
    let x = tape.term("x", 0.);
    let exp_x = (-x).apply("exp", f64::exp, f64::exp);
    let one = tape.term("1", 1.);
    let all = one / (one + exp_x);
    (x, all)
}
