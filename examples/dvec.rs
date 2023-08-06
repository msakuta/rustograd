use std::io::Write;

use rustograd::Dvec;

fn main() {
    let mut f = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    writeln!(f, "x, exp(-x^2), d exp(-x^2)/dx, d^2 exp(-x^2)/dx^2").unwrap();
    for i in -40..40 {
        let x = i as f64 / 10.;
        let d1 = Dvec::new_n(x, 1., 2);
        let d2 = &d1 * &d1;
        let d3 = -&d2;
        let d4 = d3.exp();
        // let d3 = d2.sin();
        let res = d4;
        writeln!(f, "{x}, {}, {}, {}", res[0], res[1], res[2]).unwrap();
    }
}
