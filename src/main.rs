mod term;

use term::Term;

fn main() {
    let a = Term::new(123.);
    let b = Term::new(321.);
    let c = Term::new(42.);
    let ab = &a + &b;
    let abc = &ab * &c;
    println!("a + b: {:?}", ab);
    println!("(a + b) * c: {:?}", abc);
    let ab_a = ab.derive(&a);
    println!("d(a + b) / da = {:?}", ab_a);
    let abc_a = abc.derive(&a);
    println!("d((a + b) * c) / da = {}", abc_a);
    let abc_b = abc.derive(&b);
    println!("d((a + b) * c) / db = {}", abc_b);
    let abc_c = abc.derive(&c);
    println!("d((a + b) * c) / dc = {}", abc_c);

    let d = Term::new(2.);
    let abcd = &abc / &d;
    let abcd_c = abcd.derive(&c);
    println!("d((a + b) * c / d) / dc = {}", abcd_c);

    let exp_abc = abc.exp();
    let exp_abc_c = exp_abc.derive(&c);
    println!("d(exp((a + b) * c)) / dc = {}", exp_abc_c);

    exp_abc.backprop();
    println!("exp_abc: {exp_abc:#?}");
}
