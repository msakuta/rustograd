#[macro_use]
extern crate rustograd_macro;

use rustograd_macro::rustograd;

use rustograd::Term;

fn main() {
    rustograd! {{
        let x = 123.;
        let result = 2. * x + 321.;
    }};
    let res = result.eval();
    println!("f(x): {res}");
    let grad = result.derive(&x);
    println!("df/dx: {}", grad);
    result.backprop();
    result.dot(&mut std::io::stdout()).unwrap();
}
