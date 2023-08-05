use rustograd::{Tape, TapeTerm};

use std::io::Write;

fn main() {
    let tape = Tape::new();
    let (x, all) = build_model(&tape);
    let mut file = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    writeln!(
        file,
        "x, f(x), $df/dx$ (derive), $df/dx$ (backprop), $d^2f/dx^2$ (derive2)"
    )
    .unwrap();
    for i in -40..=40 {
        let xval = i as f64 / 20. * std::f64::consts::PI;
        x.set(xval).unwrap();
        let value = all.eval();
        all.backprop().unwrap();
        let derive = all.derive(&x).unwrap();
        let derive2 = all.derive2(&x).unwrap();
        let grad = x.grad().unwrap();
        writeln!(file, "{xval}, {value}, {derive}, {grad}, {derive2}").unwrap();
    }
    x.set(0.).unwrap();
    all.eval();
    all.backprop().unwrap();
    all.dot(&mut std::io::stdout()).unwrap();
}

fn build_model(tape: &Tape) -> (TapeTerm, TapeTerm) {
    let x = tape.term("x", 0.);
    let exp_x = (-x).apply("exp", f64::exp, f64::exp, f64::exp);
    let one = tape.term("1", 1.);
    let one2 = tape.term("1", 1.);
    let all = one / (one2 + exp_x);
    (x, all)
}
