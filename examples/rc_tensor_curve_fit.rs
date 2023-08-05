//! Least squares fitting to a Gaussian distribution using gradient descent.

use rustograd::{RcTerm, Tensor};

use std::{fmt::Display, io::Write, ops::Range};

#[derive(Clone, PartialEq, Debug)]
struct MyTensor(Vec<f64>);

impl MyTensor {
    fn exp(self) -> Self {
        Self(self.0.into_iter().map(f64::exp).collect())
    }
}

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

fn main() {
    let model = build_model();

    fn truth(x: f64) -> f64 {
        0.75 * (-(x - 1.2).powf(2.) / 0.35).exp()
    }

    let samples: Vec<_> = (-40..40).map(|i| i as f64 / 10.).collect();
    let truth_data: Vec<_> = samples.iter().map(|x| truth(*x)).collect();

    let calc_loss = || {
        model.x.set(MyTensor(samples.clone())).unwrap();
        model.loss.eval().0.iter().sum::<f64>()
    };

    const RATE: f64 = 0.01;
    const INIT_MU: f64 = 0.;
    const INIT_SIGMA: f64 = 1.;
    const INIT_SCALE: f64 = 1.;

    let optimize = |mu: &mut f64, sigma: &mut f64, scale: &mut f64| {
        model.mu.set(MyTensor(vec![*mu])).unwrap();
        model.sigma.set(MyTensor(vec![*sigma])).unwrap();
        model.scale.set(MyTensor(vec![*scale])).unwrap();
        // for (&xval, &sample_y) in samples.iter().zip(truth_data.iter()) {
        model.x.set(MyTensor(samples.clone())).unwrap();
        model.sample_y.set(MyTensor(truth_data.clone())).unwrap();
        model.loss.eval();
        model.loss.backprop().unwrap();
        *mu -= RATE * model.mu.grad().0.iter().sum::<f64>();
        *sigma -= RATE * model.sigma.grad().0.iter().sum::<f64>();
        *scale -= RATE * model.scale.grad().0.iter().sum::<f64>();
        // }
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
            history.push((t, mu_val, sigma_val, scale_val));
        }
        println!(
            "i: {i}, mu: {mu_val}, sigma: {sigma_val}, scale: {scale_val}, loss: {}",
            calc_loss()
        );
    }

    let mut file = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    writeln!(file, "x, y, init_y, truth_y").unwrap();
    model.x.set(MyTensor(samples.clone())).unwrap();
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
        .zip(truth_data.iter())
        .enumerate()
    {
        let hist_string = history
            .iter()
            .map(|&(_, mu_val, sigma_val, scale_val)| {
                model.mu.set(MyTensor(vec![mu_val])).unwrap();
                model.sigma.set(MyTensor(vec![sigma_val])).unwrap();
                model.scale.set(MyTensor(vec![scale_val])).unwrap();
                model.gaussian.eval().0[i]
            })
            .fold("".to_string(), |acc, cur| acc + &format!(", {cur}"));
        writeln!(file, "{xval}, {init_y}, {truth_y}{hist_string}").unwrap();
    }

    let counter = std::cell::Cell::new(0);
    let callback = |_val| {
        let i = counter.get();
        let mut file =
            std::io::BufWriter::new(std::fs::File::create(format!("dot{i}.dot")).unwrap());
        model
            .loss
            .dot_builder()
            .show_values(false)
            .dot(&mut file)
            .unwrap();
        counter.set(i + 1);
    };

    model.x.set(MyTensor(vec![0.])).unwrap();
    model.loss.clear();
    model.loss.clear_grad();
    model.loss.eval_cb(&callback);
    model.loss.backprop_cb(&callback).unwrap();
    let mut dotfile = std::io::BufWriter::new(std::fs::File::create("graph.dot").unwrap());
    model
        .loss
        .dot_builder()
        .show_values(false)
        .dot(&mut dotfile)
        .unwrap();
}

struct Model {
    x: RcTerm<MyTensor>,
    mu: RcTerm<MyTensor>,
    sigma: RcTerm<MyTensor>,
    scale: RcTerm<MyTensor>,
    sample_y: RcTerm<MyTensor>,
    gaussian: RcTerm<MyTensor>,
    loss: RcTerm<MyTensor>,
}

fn build_model() -> Model {
    let zeros = MyTensor::default();

    let x = RcTerm::new("x", zeros.clone());
    let mu = RcTerm::new("mu", zeros.clone());
    let sigma = RcTerm::new("sigma", MyTensor::one());
    let scale = RcTerm::new("scale", MyTensor::one());
    let x_mu = &x - &mu;
    let x_mu2 = &x_mu * &x_mu;
    let x_mu2_sigma = &x_mu2 / &sigma;
    let pre_exp = &x_mu2_sigma / &sigma;
    let neg_pre_exp = -&pre_exp;
    let gaussian = &scale * &neg_pre_exp.apply("exp", MyTensor::exp, MyTensor::exp);
    let sample_y = RcTerm::new("y", zeros.clone());
    let diff = &gaussian - &sample_y;
    let loss = &diff * &diff;

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
