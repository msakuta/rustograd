use rustograd::Term;

use std::io::Write;

fn main() {
    let a = Term::new("a", 0.);
    let sin_a = a.apply("sin", f64::sin, f64::cos);

    let mut file = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    for i in -10..=10 {
        let x = i as f64 / 10. * std::f64::consts::PI;
        a.set(x).unwrap();
        sin_a.eval();
        writeln!(file, "{x}, {}, {}", sin_a.eval(), sin_a.derive(&a)).unwrap();
    }
    a.set(std::f64::consts::PI * 0.25).unwrap();
    sin_a.eval();
    sin_a.backprop();
    sin_a.dot(&mut std::io::stdout()).unwrap();
}
