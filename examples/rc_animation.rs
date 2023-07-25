//! Produces a series of .dot files to the working directory, which you can use to
//! generate a sequence of images to make an animation of forward/backpropagation.

use rustograd::RcTerm;

fn main() {
    let a = RcTerm::new("a", 123.);
    let b = RcTerm::new("b", 321.);
    let c = RcTerm::new("c", 42.);
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

    abc.clear_grad();
    abc.clear();

    let counter = std::cell::Cell::new(0);
    let callback = |_val| {
        let i = counter.get();
        let mut file =
            std::io::BufWriter::new(std::fs::File::create(format!("dot{i}.dot")).unwrap());
        abc.dot(&mut file).unwrap();
        counter.set(i + 1);
    };
    abc.eval_cb(&callback);

    abc.backprop_cb(&callback);
    abc.dot(&mut std::io::stdout()).unwrap();
}
