use rustograd::RcTerm;

use std::io::Write;

fn main() {
    let (a, all) = build_model();
    let mut file = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    for i in -40..=40 {
        let x = i as f64 / 20. * std::f64::consts::PI;
        a.set(x).unwrap();
        writeln!(file, "{x}, {}, {}", all.eval(), all.derive(&a)).unwrap();
    }
    all.backprop();
    all.dot(&mut std::io::stdout()).unwrap();
}

fn build_model() -> (RcTerm, RcTerm) {
    let a = RcTerm::new("a", 0.);
    let sin_a = a.apply("sin", f64::sin, f64::cos);
    let ten = RcTerm::new("5", 5.);
    let b = &a * &ten;
    let c = RcTerm::new("c", 0.2);
    let sin_b = b.apply("sin", f64::sin, f64::cos);
    let c_sin_b = &c * &sin_b;
    let all = &sin_a + &c_sin_b;
    (a, all)
}
