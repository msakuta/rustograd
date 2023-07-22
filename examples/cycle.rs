use rustograd::Term;

fn main() {
    let a = Term::new("a".to_string(), 1.);
    let b = Term::new("b".to_string(), 3.);
    let c = Term::new("c".to_string(), 5.);
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
