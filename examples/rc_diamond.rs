//! Dependency graph in diamond shape. It evaluates the same term twice, so the derivative should add up.

use rustograd_macro::rustograd;

fn main() {
    rustograd! {{
        let a = 1.;
        let a2 = -a;
        let b = 3.;
        let c = 5.;
        let ab = a2 + b;
        let ac = a2 + c;
        let abac = ab + ac;
    }}

    let counter = std::cell::Cell::new(0);
    let callback = |term| {
        let i = counter.get();
        let mut file =
            std::io::BufWriter::new(std::fs::File::create(format!("dot{i}.dot")).unwrap());
        abac.dot(&mut file, Some(&term)).unwrap();
        counter.set(i + 1);
    };

    abac.eval_cb(&callback);
    abac.backprop_cb(&callback);
    println!("abac: {}", abac.grad());
    println!("a: {}", a.grad());
    println!("b: {}", b.grad());
    println!("c: {}", c.grad());
    abac.dot(&mut std::io::stdout(), None).unwrap();
}
