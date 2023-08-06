use std::{fmt::Display, io::Write, ops::Range};

use rustograd::{Dvec, Tensor};

#[derive(Clone)]
struct MyTensor(Vec<f64>);

const XRANGE: Range<i32> = -40..40;
const ELEMS: usize = (XRANGE.end - XRANGE.start) as usize;

impl MyTensor {
    fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self(self.0.iter().map(|&v| f(v)).collect())
    }
}

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
    let mut f = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    writeln!(
        f,
        "x, exp(-x^2), d exp(-x^2)/dx, d^2 exp(-x^2)/dx^2, d^3 exp(-x^2)/dx^3"
    )
    .unwrap();
    let xs = MyTensor(XRANGE.map(|i| i as f64 / 10.).collect());
    let d1 = Dvec::new_n(xs.clone(), MyTensor::one(), 3);
    let d2 = &d1 * &d1;
    let d3 = -&d2;
    let d4 = d3.apply(|x, _| x.map(f64::exp));
    // let d4 = d1.apply(|x, n| {
    //     match n % 4 {
    //         0 => x.map(f64::sin),
    //         1 => x.map(f64::cos),
    //         2 => -x.map(f64::sin),
    //         3 => -x.map(f64::cos),
    //         _ => unreachable!(),
    //     }
    // });
    let res = d4;

    for ((((x, y), dy), d2y), d3y) in
        xs.0.iter()
            .zip(res[0].0.iter())
            .zip(res[1].0.iter())
            .zip(res[2].0.iter())
            .zip(res[3].0.iter())
    {
        writeln!(f, "{x}, {}, {}, {}, {}", y, dy, d2y, d3y).unwrap();
    }
}
