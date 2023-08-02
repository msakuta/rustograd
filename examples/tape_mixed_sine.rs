use rustograd::{Tape, TapeTerm};

use std::io::Write;

fn main() {
    let tape = Tape::new();
    let (a, all) = build_model(&tape);
    let mut file = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    for i in -40..=40 {
        let x = i as f64 / 20. * std::f64::consts::PI;
        a.set(x).unwrap();
        let value = all.eval();
        all.backprop().unwrap();
        let grad = a.grad().unwrap();
        writeln!(file, "{x}, {value}, {}, {grad}", all.derive(&a).unwrap()).unwrap();
    }
    a.set(0.).unwrap();
    all.backprop().unwrap();
    all.dot(&mut std::io::stdout()).unwrap();
}

fn build_model(tape: &Tape) -> (TapeTerm, TapeTerm) {
    let a = tape.term("a", 0.);
    let sin_a = a.apply("sin", f64::sin, f64::cos);
    let b = a * tape.term("5", 5.);
    let c = tape.term("c", 0.2);
    let sin_b = b.apply("sin", f64::sin, f64::cos);
    let all = sin_a + c * sin_b;
    (a, all)
}
