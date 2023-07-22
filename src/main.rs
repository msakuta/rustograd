use std::ops::{Add, Mul};

#[derive(Clone, Debug)]
enum Term<'a> {
    Value(f64),
    Add(&'a Term<'a>, &'a Term<'a>),
    Mul(&'a Term<'a>, &'a Term<'a>),
}

impl<'a> Add for &'a Term<'a> {
    type Output = Box<Term<'a>>;
    fn add(self, rhs: Self) -> Self::Output {
        Box::new(Term::Add(self, rhs))
    }
}

impl<'a> Mul for &'a Term<'a> {
    type Output = Box<Term<'a>>;
    fn mul(self, rhs: Self) -> Self::Output {
        Box::new(Term::Mul(self, rhs))
    }
}

impl<'a> Term<'a> {
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
    let a = Term::Value(123.);
    let b = Term::Value(321.);
    let c = Term::Value(42.);
    let ab = &a + &b;
    let abc = &ab as &Term * &c;
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
