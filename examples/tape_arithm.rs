use rustograd::Tape;

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 123.);
    let b = tape.term("b", 321.);
    let c = tape.term("c", 42.);
    let ab = a + b;
    let abc = ab * c;
    println!("a + b = {}", ab.eval());
    println!("(a + b) * c = {}", abc.eval());
    let ab_a = ab.derive(&a).unwrap();
    println!("d(a + b) / da = {}", ab_a);
    let abc_a = abc.derive(&a).unwrap();
    println!("d((a + b) * c) / da = {}", abc_a);
    let abc_b = abc.derive(&b).unwrap();
    println!("d((a + b) * c) / db = {}", abc_b);
    let abc_c = abc.derive(&c).unwrap();
    println!("d((a + b) * c) / dc = {}", abc_c);

    let d = tape.term("d", 2.);
    let abcd = abc / d;
    abcd.eval();
    let abcd_c = abcd.derive(&c).unwrap();
    println!("d((a + b) * c / d) / dc = {}", abcd_c);
}
