use rustograd::{HStack, Tape, TapeTerm, Tensor};

use std::{fmt::Display, io::Write, ops::Range};

#[derive(Clone, Debug)]
struct MyTensor(Vec<f64>);

const XRANGE0: Range<i32> = -40..0;
const XRANGE1: Range<i32> = 0..40;

fn broadcast_binop(lhs: MyTensor, rhs: MyTensor, op: impl Fn(f64, f64) -> f64) -> MyTensor {
    // Broadcasting rules
    MyTensor(if lhs.0.len() == 1 {
        let lhs = lhs.0[0];
        rhs.0.into_iter().map(|rhs| op(lhs, rhs)).collect()
    } else if rhs.0.len() == 1 {
        let rhs = rhs.0[0];
        lhs.0.into_iter().map(|lhs| op(lhs, rhs)).collect()
    } else {
        assert_eq!(lhs.0.len(), rhs.0.len());
        lhs.0
            .into_iter()
            .zip(rhs.0.into_iter())
            .map(|(lhs, rhs)| op(lhs, rhs))
            .collect()
    })
}

fn broadcast_binassign(lhs: &mut MyTensor, rhs: MyTensor, op: impl Fn(f64, f64) -> f64) {
    // Broadcasting rules
    if lhs.0.len() == 1 {
        let lhs_val = lhs.0[0];
        lhs.0 = rhs.0.into_iter().map(|rhs| op(lhs_val, rhs)).collect();
    } else if rhs.0.len() == 1 {
        let rhs = rhs.0[0];
        lhs.0.iter_mut().for_each(|lhs| *lhs = op(*lhs, rhs));
    } else {
        lhs.0
            .iter_mut()
            .zip(rhs.0.into_iter())
            .for_each(|(lhs, rhs)| *lhs = op(*lhs, rhs));
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
        Self(vec![0.])
    }
}

impl std::ops::Add for MyTensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        broadcast_binop(self, rhs, f64::add)
    }
}

impl std::ops::AddAssign for MyTensor {
    fn add_assign(&mut self, rhs: Self) {
        broadcast_binassign(self, rhs, |lhs, rhs| lhs + rhs);
    }
}

impl std::ops::Sub for MyTensor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        broadcast_binop(self, rhs, f64::sub)
    }
}

impl std::ops::SubAssign for MyTensor {
    fn sub_assign(&mut self, rhs: Self) {
        broadcast_binassign(self, rhs, |lhs, rhs| lhs - rhs);
    }
}

impl std::ops::Mul for MyTensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        broadcast_binop(self, rhs, f64::mul)
    }
}

// scale
impl std::ops::Mul<f64> for MyTensor {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Self(self.0.into_iter().map(|lhs| lhs * rhs).collect())
    }
}

impl std::ops::Div for MyTensor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        broadcast_binop(self, rhs, f64::div)
    }
}

impl std::ops::Neg for MyTensor {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(self.0.into_iter().map(|lhs| -lhs).collect())
    }
}

impl HStack for MyTensor {
    fn hstack(mut self, mut rhs: Self) -> Self {
        self.0.append(&mut rhs.0);
        self
    }
}

impl Tensor for MyTensor {
    fn one() -> Self {
        Self(vec![1.])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|v| *v == 0.)
    }

    fn hstack(self, rhs: Self) -> Option<Self> {
        Some(<Self as HStack>::hstack(self, rhs))
    }
}

fn main() {
    let tape = Tape::<MyTensor>::new();
    let (x0, x1, x, all) = build_model(&tape);
    let mut file = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    let xs0 = MyTensor(
        XRANGE0
            .map(|i| i as f64 / 20. * std::f64::consts::PI)
            .collect(),
    );
    let xs1 = MyTensor(
        XRANGE1
            .map(|i| i as f64 / 20. * std::f64::consts::PI)
            .collect(),
    );
    x0.set(xs0.clone()).unwrap();
    x1.set(xs1.clone()).unwrap();
    let value = all.eval();
    println!("value: {}", x.value().unwrap().0.len());
    // all.backprop().unwrap();
    let derive0 = all.derive(&x0).unwrap();
    let derive1 = all.derive(&x1).unwrap();
    println!(
        "value: {} derive: {}",
        all.value().unwrap().0.len(),
        derive0.0.len()
    );
    // let grad0 = x0.grad().unwrap();
    writeln!(file, "x, f(x), $df/dx$ (derive), $df/dx$ (backprop)").unwrap();
    // for (((xval, &value), derive), grad) in XRANGE0
    for ((&xval, &value), derive) in xs0
        .0
        .iter()
        .chain(xs1.0.iter())
        .zip(value.0.iter())
        .zip(derive0.0.iter().chain(derive1.0.iter()))
    // .zip(grad0.0.iter())
    {
        // writeln!(file, "{xval}, {value}, {derive}, {grad}").unwrap();
        writeln!(file, "{xval}, {value}, {derive}").unwrap();
    }
    x0.set(MyTensor::default()).unwrap();
    x1.set(MyTensor::default()).unwrap();
    all.eval();
    // all.backprop().unwrap();
    all.dot(&mut std::io::stdout()).unwrap();
}

fn my_exp(x: MyTensor) -> MyTensor {
    MyTensor(x.0.into_iter().map(|x| x.exp()).collect())
}

fn build_model(
    tape: &Tape<MyTensor>,
) -> (
    TapeTerm<MyTensor>,
    TapeTerm<MyTensor>,
    TapeTerm<MyTensor>,
    TapeTerm<MyTensor>,
) {
    let x0 = tape.term("x0", MyTensor::default());
    let x1 = tape.term("x1", MyTensor::default());
    let x = x0.apply_binary(
        "hstack",
        x1,
        |lhs, rhs| <MyTensor as HStack>::hstack(lhs, rhs),
        |lhs, rhs| (lhs, rhs),
    );
    // let x = x0.hstack(x1);
    let exp_x = (-x).apply("exp", my_exp, my_exp);
    let one = tape.term("1", MyTensor::one());
    let one2 = tape.term("1", MyTensor::one());
    let one_add = one2 + exp_x;
    let all = one / one_add;
    (x0, x1, exp_x, all)
}
