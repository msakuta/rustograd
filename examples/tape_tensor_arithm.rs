use std::fmt::Display;

use rustograd::{Tape, TapeTerm, Tensor};

const ELEMS: usize = 10;

#[derive(Clone)]
struct MyTensor(Vec<f64>);

impl Display for MyTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for item in &self.0 {
            write!(f, "{item}, ")?;
        }
        Ok(())
    }
}

impl Default for MyTensor {
    fn default() -> Self {
        Self(vec![0.; ELEMS])
    }
}

impl std::ops::Add for MyTensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(
            self.0
                .into_iter()
                .zip(rhs.0.into_iter())
                .map(|(lhs, rhs)| lhs + rhs)
                .collect(),
        )
    }
}

impl std::ops::AddAssign for MyTensor {
    fn add_assign(&mut self, rhs: Self) {
        for (rhs, lhs) in self.0.iter_mut().zip(rhs.0.into_iter()) {
            *rhs += lhs;
        }
    }
}
impl std::ops::Sub for MyTensor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(
            self.0
                .into_iter()
                .zip(rhs.0.into_iter())
                .map(|(lhs, rhs)| lhs - rhs)
                .collect(),
        )
    }
}

impl std::ops::Mul for MyTensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(
            self.0
                .into_iter()
                .zip(rhs.0.into_iter())
                .map(|(lhs, rhs)| lhs * rhs)
                .collect(),
        )
    }
}

impl std::ops::Div for MyTensor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(
            self.0
                .into_iter()
                .zip(rhs.0.into_iter())
                .map(|(lhs, rhs)| lhs / rhs)
                .collect(),
        )
    }
}

impl std::ops::Neg for MyTensor {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(self.0.into_iter().map(|lhs| -lhs).collect())
    }
}

impl Tensor for MyTensor {
    fn one() -> Self {
        Self(vec![1.; ELEMS])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|v| *v == 0.)
    }
}

fn main() {
    let tape = Tape::<MyTensor>::new();
    let a = tape.term("a", MyTensor(vec![123.; ELEMS]));
    let b = tape.term("b", MyTensor(vec![321.; ELEMS]));
    let c = tape.term("c", MyTensor(vec![42.; ELEMS]));
    let ab = a + b;
    let abc = ab * c;
    println!("a + b = {}", ab.eval());
    println!("(a + b) * c = {}", abc.eval());
    let ab_a = ab.derive(&a);
    println!("d(a + b) / da = {}", ab_a);
    let abc_a = abc.derive(&a);
    println!("d((a + b) * c) / da = {}", abc_a);
    let abc_b = abc.derive(&b);
    println!("d((a + b) * c) / db = {}", abc_b);
    let abc_c = abc.derive(&c);
    println!("d((a + b) * c) / dc = {}", abc_c);

    let d = tape.term("d", MyTensor(vec![2.; ELEMS]));
    let abcd = abc / d;
    let abcd_c = abcd.derive(&c);
    println!("d((a + b) * c / d) / dc = {}", abcd_c);
}
