//! Dependency graph in diamond shape. It evaluates the same term twice, so the derivative should add up.

use rustograd::Tape;

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 1.);
    let a2 = -a;
    let b = tape.term("b", 3.);
    let c = tape.term("c", 5.);
    let ab = a2 + b;
    let ac = a2 + c;
    let abac = ab + ac;

    let counter = std::cell::Cell::new(0);
    let callback = |nodes: &_, idx| {
        let i = counter.get();
        let mut file =
            std::io::BufWriter::new(std::fs::File::create(format!("dot{i}.dot")).unwrap());
        abac.dot_builder()
            .show_values(true)
            .highlights(idx)
            .dot_borrowed(nodes, &mut file)
            .unwrap();
        counter.set(i + 1);
    };

    abac.eval_cb(&callback);
    abac.backprop_cb(&callback).unwrap();
    println!("abac: {}", abac.grad().unwrap());
    println!("a: {}", a.grad().unwrap());
    println!("b: {}", b.grad().unwrap());
    println!("c: {}", c.grad().unwrap());
    abac.dot_builder()
        .show_values(true)
        .dot(&mut std::io::stdout())
        .unwrap();
}
