use rustograd::Term;

fn main() {
    let a = Term::new(1.);
    let b = Term::new(3.);
    let c = Term::new(5.);
    let ab = &a + &b;
    let ac = &a + &c;
    let abac = &ab + &ac;

    abac.backprop();
    println!("abac: {abac:#?}");
    println!("a: {a:?}");
    println!("b: {b:?}");
    println!("c: {c:?}");
}
