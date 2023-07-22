use rustograd::Term;

use std::io::Write;

fn main() {
    let mut file = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    for i in -40..=40 {
        let x = i as f64 / 20. * std::f64::consts::PI;
        run_model(x, |all, a| {
            writeln!(file, "{x}, {}, {}", all.eval(), all.derive(&a)).unwrap();
        });
    }
    run_model(0., |all, _| {
        all.backprop();
        all.dot(&mut std::io::stdout()).unwrap();
    });
}

fn run_model(a_val: f64, f: impl FnOnce(&Term, &Term)) {
    let a = Term::new("a", a_val);
    let sin_a = a.apply("sin", f64::sin, f64::cos);
    let ten = Term::new("5", 5.);
    let b = &a * &ten;
    let c = Term::new("c", 0.2);
    let sin_b = b.apply("sin", f64::sin, f64::cos);
    let c_sin_b = &c * &sin_b;
    let all = &sin_a + &c_sin_b;

    f(&all, &a);
}
