//! Dependency graph in diamond shape. It evaluates the same term twice, so the derivative should add up.

use rustograd::Term;

fn main() {
    let a = Term::new("a", 1.);
    let b = Term::new("b", 3.);
    let c = Term::new("c", 5.);
    let ab = &a + &b;
    let ac = &a + &c;
    let abac = &ab + &ac;

    abac.backprop();
    println!("abac: {abac:#?}");
    println!("a: {a:?}");
    println!("b: {b:?}");
    println!("c: {c:?}");
    abac.dot(&mut std::io::stdout()).unwrap();
}
