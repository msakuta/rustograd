use rustograd::{Tape, TapeTerm, Tensor};

use std::{fmt::Display, io::Write, ops::Range};

#[derive(Clone)]
struct MyTensor(Vec<f64>);

const XRANGE: Range<i32> = -40..40;
const ELEMS: usize = (XRANGE.end - XRANGE.start) as usize;

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
    let (x, all) = build_model(&tape);
    let mut file = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    let xs = MyTensor(
        XRANGE
            .map(|i| i as f64 / 20. * std::f64::consts::PI)
            .collect(),
    );
    x.set(xs).unwrap();
    let value = all.eval();
    all.backprop().unwrap();
    let derive = all.derive(&x).unwrap();
    let grad = x.grad().unwrap();
    writeln!(file, "x, f(x), $df/dx$ (derive), $df/dx$ (backprop)").unwrap();
    for (((xval, &value), derive), grad) in XRANGE
        .zip(value.0.iter())
        .zip(derive.0.iter())
        .zip(grad.0.iter())
    {
        writeln!(file, "{xval}, {value}, {derive}, {grad}").unwrap();
    }
    x.set(MyTensor::default()).unwrap();
    all.eval();
    all.backprop().unwrap();
    all.dot(&mut std::io::stdout()).unwrap();
}

fn my_exp(x: MyTensor) -> MyTensor {
    MyTensor(x.0.into_iter().map(|x| x.exp()).collect())
}

fn build_model(tape: &Tape<MyTensor>) -> (TapeTerm<MyTensor>, TapeTerm<MyTensor>) {
    let x = tape.term("x", MyTensor::default());
    let exp_x = (-x).apply("exp", my_exp, my_exp);
    let one = tape.term("1", MyTensor::one());
    let one2 = tape.term("1", MyTensor::one());
    let all = one / (one2 + exp_x);
    (x, all)
}
