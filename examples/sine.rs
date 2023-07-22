use rustograd::Term;

fn main() {
    for i in -10..=10 {
        let x = i as f64 / 10. * std::f64::consts::PI;
        run_model(x);
    }
}

fn run_model(a_val: f64) {
    let a = Term::new(a_val);
    let sin_a = a.apply(f64::sin, f64::cos);

    println!("[{a_val}, {}, {}],", sin_a.eval(), sin_a.derive(&a));
}
