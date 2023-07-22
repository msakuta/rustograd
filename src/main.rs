use std::ops::{Add, Mul};

#[derive(Clone, Debug)]
enum TermInt<'a> {
    Value(f64),
    Add(&'a Term<'a>, &'a Term<'a>),
    Mul(&'a Term<'a>, &'a Term<'a>),
}

type Term<'a> = Box<TermInt<'a>>;

impl<'a> Add for &'a Term<'a> {
    type Output = Term<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        Box::new(TermInt::Add(self, rhs))
    }
}

impl<'a> Mul for &'a Term<'a> {
    type Output = Term<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        Box::new(TermInt::Mul(self, rhs))
    }
}

impl<'a> TermInt<'a> {
    fn new(val: f64) -> Term<'a> {
        Box::new(TermInt::Value(val))
    }
    fn derive(&self, var: &Self) -> f64 {
        if self as *const _ == var as *const _ {
            1.
        } else {
            match self {
                Self::Value(_) => 0.,
                Self::Add(lhs, rhs) => lhs.derive(var) + rhs.derive(var),
                Self::Mul(lhs, rhs) => {
                    let dlhs = lhs.derive(var);
                    let drhs = rhs.derive(var);
                    dlhs * rhs.eval() + lhs.eval() * drhs
                }
            }
        }
    }

    fn eval(&self) -> f64 {
        match self {
            Self::Value(val) => *val,
            Self::Add(lhs, rhs) => lhs.eval() + rhs.eval(),
            Self::Mul(lhs, rhs) => lhs.eval() * rhs.eval(),
        }
    }
}

fn main() {
    let a = TermInt::new(123.);
    let b = TermInt::new(321.);
    let c = TermInt::new(42.);
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
}
