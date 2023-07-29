use rustograd::RcTerm;

use std::io::Write;

fn main() {
    let (x, all) = build_model();
    let mut file = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    for i in -40..=40 {
        let xval = i as f64 / 20. * std::f64::consts::PI;
        x.set(xval).unwrap();
        let value = all.eval();
        all.backprop();
        let derive = all.derive(&x);
        let grad = x.grad();
        writeln!(file, "{xval}, {value}, {derive}, {grad}").unwrap();
    }
    x.set(0.).unwrap();
    all.eval();
    all.backprop();
    all.dot(&mut std::io::stdout()).unwrap();
}

fn build_model() -> (RcTerm, RcTerm) {
    let x = RcTerm::new("x", 0.);
    let exp_x = (-&x).apply("exp", f64::exp, f64::exp);
    let one = RcTerm::new("1", 1.);
    let one2 = RcTerm::new("1", 1.);
    let one2_exp_x = &one2 + &exp_x;
    let all = &one / &one2_exp_x;
    (x, all)
}
