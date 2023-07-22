use rustograd::Term;

#[test]
fn main() {
    let a = Term::new(1.);
    let b = Term::new(3.);
    let c = Term::new(5.);
    let ab = &a + &b;
    let ac = &a + &c;
    let abac = &ab + &ac;

    abac.backprop();
    println!("abac: {abac:#?}");
    assert_eq!(a.grad(), 2.);
    assert_eq!(b.grad(), 1.);
    assert_eq!(c.grad(), 1.);
}
