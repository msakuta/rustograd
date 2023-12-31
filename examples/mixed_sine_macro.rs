//! An example with procedural macro.
//! Requires a feature flag `macro`.
//! Use `cargo r --features macro --example mixed_sine_macro`

use rustograd::RcTerm;
use rustograd_macro::rustograd;

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
    rustograd! {{
        let a = 0.;
        let all = sin(a) + 0.2 * sin(a * 5.);
    }}
    (a, all)
}

fn sin(x: f64) -> f64 {
    x.sin()
}

fn sin_derive(x: f64) -> f64 {
    x.cos()
}
