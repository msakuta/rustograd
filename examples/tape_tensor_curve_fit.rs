//! Least squares fitting to a Gaussian distribution using gradient descent.

use rustograd::{Tape, TapeTerm, Tensor};

use std::{fmt::Display, io::Write, ops::Range};

#[derive(Clone, PartialEq, Debug)]
struct MyTensor(Vec<f64>);

const XRANGE: Range<i32> = -40..40;

fn broadcast_binop(lhs: MyTensor, rhs: MyTensor, op: impl Fn(f64, f64) -> f64) -> MyTensor {
    // Broadcasting rules
    MyTensor(if lhs.0.len() == 1 {
        let lhs = lhs.0[0];
        rhs.0.into_iter().map(|rhs| op(lhs, rhs)).collect()
    } else if rhs.0.len() == 1 {
        let rhs = rhs.0[0];
        lhs.0.into_iter().map(|lhs| op(lhs, rhs)).collect()
    } else {
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
        Self(vec![0.; 1])
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

impl Tensor for MyTensor {
    fn one() -> Self {
        Self(vec![1.; 1])
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|v| *v == 0.)
    }
}

#[test]
fn test_tensor_broadcasting() {
    assert_eq!(
        MyTensor(vec![1.; 1]) + MyTensor(vec![10.; 3]),
        MyTensor(vec![11.; 3])
    );
    assert_eq!(
        MyTensor(vec![5.; 3]) * MyTensor(vec![10.; 1]),
        MyTensor(vec![50.; 3])
    );
}

fn main() {
    let tape = Tape::new();
    let model = build_model(&tape);

    fn truth(x: f64) -> f64 {
        0.75 * (-(x - 1.2).powf(2.) / 0.35).exp()
    }

    let samples: Vec<_> = XRANGE.map(|i| i as f64 / 10.).collect();
    let xs = MyTensor(samples.clone());
    let truth_data = MyTensor(samples.iter().map(|x| truth(*x)).collect());

    let calc_loss = || {
        model.x.set(xs.clone()).unwrap();
        model.loss.eval().0.into_iter().sum::<f64>()
    };

    const RATE: f64 = 0.01;
    const INIT_MU: f64 = 0.;
    const INIT_SIGMA: f64 = 1.;
    const INIT_SCALE: f64 = 1.;

    let optimize = |mu: &mut f64, sigma: &mut f64, scale: &mut f64| {
        model.mu.set(MyTensor(vec![*mu])).unwrap();
        model.sigma.set(MyTensor(vec![*sigma])).unwrap();
        model.scale.set(MyTensor(vec![*scale])).unwrap();
        // for (&xval, &sample_y) in samples.iter().zip(truth_data.iter())
        {
            model.x.set(xs.clone()).unwrap();
            model.sample_y.set(truth_data.clone()).unwrap();
            model.loss.eval();
            model.loss.backprop();
            // println!("x: {}, sample_y: {}, dmu: {}", xs.0.len(), truth_data.0.len(), model.mu.grad().0.iter().sum::<f64>());
            *mu -= model.mu.grad().0.iter().sum::<f64>() * RATE;
            *sigma -= model.sigma.grad().0.iter().sum::<f64>() * RATE;
            *scale -= model.scale.grad().0.iter().sum::<f64>() * RATE;
        }
    };

    let mut mu_val = INIT_MU;
    let mut sigma_val = INIT_SIGMA;
    let mut scale_val = INIT_SCALE;
    let mut history = vec![];
    for i in 0..100 {
        optimize(&mut mu_val, &mut sigma_val, &mut scale_val);
        let t = i as f64;
        if history
            .last()
            .map(|last: &(f64, _, _, _)| last.0 * 1.2 < t)
            .unwrap_or(true)
        {
            history.push((t, mu_val.clone(), sigma_val.clone(), scale_val.clone()));
        }
        println!(
            "i: {i}, mu: {mu_val}, sigma: {sigma_val}, scale: {scale_val}, loss: {}",
            calc_loss()
        );
    }

    let mut file = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    writeln!(file, "x, init_y, truth_y").unwrap();
    model.x.set(xs).unwrap();
    model.mu.set(MyTensor(vec![mu_val])).unwrap();
    model.sigma.set(MyTensor(vec![sigma_val])).unwrap();
    model.scale.set(MyTensor(vec![scale_val])).unwrap();
    model.mu.set(MyTensor(vec![INIT_MU])).unwrap();
    model.sigma.set(MyTensor(vec![INIT_SIGMA])).unwrap();
    model.scale.set(MyTensor(vec![INIT_SCALE])).unwrap();
    let init_value = model.gaussian.eval();
    for (i, ((&xval, &init_y), &truth_y)) in samples
        .iter()
        .zip(init_value.0.iter())
        .zip(truth_data.0.iter())
        .enumerate()
    {
        let hist_string = history
            .iter()
            .map(|(_, mu_val, sigma_val, scale_val)| {
                model.mu.set(MyTensor(vec![*mu_val])).unwrap();
                model.sigma.set(MyTensor(vec![*sigma_val])).unwrap();
                model.scale.set(MyTensor(vec![*scale_val])).unwrap();
                model.gaussian.eval().0[i]
            })
            .fold("".to_string(), |acc, cur| acc + &format!(", {cur}"));
        writeln!(file, "{xval}, {init_y}, {truth_y}{hist_string}").unwrap();
    }
    model.x.set(MyTensor::default()).unwrap();
    model.gaussian.eval();
    model.gaussian.backprop();
    let mut dotfile = std::io::BufWriter::new(std::fs::File::create("graph.dot").unwrap());
    model.gaussian.dot(&mut dotfile, false).unwrap();
}

struct Model<'a> {
    x: TapeTerm<'a, MyTensor>,
    mu: TapeTerm<'a, MyTensor>,
    sigma: TapeTerm<'a, MyTensor>,
    scale: TapeTerm<'a, MyTensor>,
    sample_y: TapeTerm<'a, MyTensor>,
    gaussian: TapeTerm<'a, MyTensor>,
    loss: TapeTerm<'a, MyTensor>,
}

fn my_exp(x: MyTensor) -> MyTensor {
    MyTensor(x.0.into_iter().map(|x| x.exp()).collect())
}

fn build_model(tape: &Tape<MyTensor>) -> Model {
    let x = tape.term("x", MyTensor::default());
    let mu = tape.term("mu", MyTensor::default());
    let sigma = tape.term("sigma", MyTensor::one());
    let scale = tape.term("scale", MyTensor::one());
    let x_mu = x - mu;
    let gaussian = scale * (-(x_mu * x_mu) / sigma / sigma).apply("exp", my_exp, my_exp);
    let sample_y = tape.term("y", MyTensor::default());
    let diff = gaussian - sample_y;
    let loss = diff * diff;
    Model {
        x,
        mu,
        sigma,
        scale,
        gaussian,
        sample_y,
        loss,
    }
}