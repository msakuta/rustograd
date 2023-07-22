use rustograd::RcTerm;

struct Model {
    a: RcTerm,
    ab: RcTerm,
}

fn model() -> Model {
    let a = RcTerm::new("a", 1.);
    let b = RcTerm::new("b", 2.);
    let ab = &a * &b;
    Model { a, ab }
}

fn main() {
    let Model { a, ab } = model();
    println!("a: {}, ab: {}, d(ab)/da: {}", a.eval(), ab.eval(), ab.derive(&a));
}