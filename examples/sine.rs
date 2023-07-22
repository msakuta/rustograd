use rustograd::Term;

fn main() {
    let a = Term::new("a", 0.);
    let sin_a = a.apply("sin", f64::sin, f64::cos);

    for i in -10..=10 {
        let x = i as f64 / 10. * std::f64::consts::PI;
        a.set(x).unwrap();
        sin_a.eval();
        println!("[{x}, {}, {}],", sin_a.eval(), sin_a.derive(&a));
    }
}
