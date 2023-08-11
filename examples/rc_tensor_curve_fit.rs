//! Least squares fitting to a Gaussian distribution using gradient descent.

use rustograd::{RcTerm, Tensor, UnaryFn};

use std::{fmt::Display, io::Write, ops::Range};

#[derive(Clone, PartialEq, Debug)]
struct MyTensor(Vec<f64>);

impl MyTensor {
    fn sum(self) -> Self {
        MyTensor(vec![self.0.iter().sum()])
    }

    fn exp(self) -> Self {
        Self(self.0.into_iter().map(f64::exp).collect())
    }
}

const XRANGE: Range<i32> = -40..40;

fn tensor_binop(lhs: MyTensor, rhs: MyTensor, op: impl Fn(f64, f64) -> f64) -> MyTensor {
    assert_eq!(lhs.0.len(), rhs.0.len());
    MyTensor(
        lhs.0
            .into_iter()
            .zip(rhs.0.into_iter())
            .map(|(lhs, rhs)| op(lhs, rhs))
            .collect(),
    )
}

fn tensor_binassign(lhs: &mut MyTensor, rhs: MyTensor, op: impl Fn(f64, f64) -> f64) {
    assert_eq!(lhs.0.len(), rhs.0.len());
    lhs.0
        .iter_mut()
        .zip(rhs.0.into_iter())
        .for_each(|(lhs, rhs)| *lhs = op(*lhs, rhs));
}

impl Display for MyTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for item in &self.0 {
            write!(f, "{item}, ")?;
        }
        // write!(f, "[{}]", self.0.len())?;
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
        tensor_binop(self, rhs, f64::add)
    }
}

impl std::ops::AddAssign for MyTensor {
    fn add_assign(&mut self, rhs: Self) {
        tensor_binassign(self, rhs, |lhs, rhs| lhs + rhs);
    }
}

impl std::ops::Sub for MyTensor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        tensor_binop(self, rhs, f64::sub)
    }
}

impl std::ops::SubAssign for MyTensor {
    fn sub_assign(&mut self, rhs: Self) {
        tensor_binassign(self, rhs, |lhs, rhs| lhs - rhs);
    }
}

impl std::ops::Mul for MyTensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        tensor_binop(self, rhs, f64::mul)
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
        tensor_binop(self, rhs, f64::div)
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

    let samples: Vec<_> = XRANGE.map(|i| i as f64 / 10.).collect();
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
        model.x.set(MyTensor(samples.clone())).unwrap();
        model.sample_y.set(MyTensor(truth_data.clone())).unwrap();
        model.loss.eval();
        model.loss.backprop().unwrap();
        *mu -= RATE * model.mu.grad().0[0];
        *sigma -= RATE * model.sigma.grad().0[0];
        *scale -= RATE * model.scale.grad().0[0];
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
            .vertical(true)
            .show_values(false)
            .dot(&mut file)
            .unwrap();
        counter.set(i + 1);
    };

    model.loss.clear();
    model.loss.clear_grad();
    model.x.set(MyTensor(samples)).unwrap();
    model.loss.eval_cb(&callback);
    model.loss.backprop_cb(&callback).unwrap();
    let mut dotfile = std::io::BufWriter::new(std::fs::File::create("graph.dot").unwrap());
    model
        .loss
        .dot_builder()
        .vertical(true)
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

fn bcast(v: MyTensor) -> MyTensor {
    // println!("Broadcasting {} to {}", v.0[0], XRANGE.len());
    MyTensor(XRANGE.map(|_| v.0[0]).collect())
}

fn bcast1(_: MyTensor) -> MyTensor {
    // println!("Broadcasting {} to {}", v.0[0], XRANGE.len());
    MyTensor(vec![1.; XRANGE.len()])
}

fn distribute(v: MyTensor) -> MyTensor {
    // println!("Ditributing {} to {}", v.0[0], XRANGE.len());
    MyTensor(XRANGE.map(|_| v.0[0] / XRANGE.len() as f64).collect())
}

fn collapse(v: MyTensor) -> MyTensor {
    // println!("Broadcasting {} to {}", v.0[0], XRANGE.len());
    MyTensor(vec![v.0.len() as f64])
}

struct Broadcaster;

impl UnaryFn<MyTensor> for Broadcaster {
    fn name(&self) -> String {
        "bcast".to_string()
    }
    fn f(&self, data: MyTensor) -> MyTensor {
        bcast(data)
    }
    fn grad(&self, data: MyTensor) -> MyTensor {
        bcast1(data)
    }
    fn t(&self, data: MyTensor) -> MyTensor {
        MyTensor::sum(data)
    }
}

struct Summer;

impl UnaryFn<MyTensor> for Summer {
    fn name(&self) -> String {
        "sum".to_string()
    }
    fn f(&self, data: MyTensor) -> MyTensor {
        MyTensor::sum(data)
    }
    fn grad(&self, data: MyTensor) -> MyTensor {
        collapse(data)
    }
    fn t(&self, data: MyTensor) -> MyTensor {
        distribute(data)
    }
}

fn build_model() -> Model {
    let zeros = MyTensor::default();

    let x = RcTerm::new("x", zeros.clone());
    let mu = RcTerm::new("mu", zeros.clone());
    let v_mu = mu.apply_t(Box::new(Broadcaster));
    let sigma = RcTerm::new("sigma", MyTensor::one());
    let sigma2 = &sigma * &sigma;
    let v_sigma2 = sigma2.apply_t(Box::new(Broadcaster));
    let scale = RcTerm::new("scale", MyTensor::one());
    let v_scale = scale.apply_t(Box::new(Broadcaster));
    let x_mu = &x - &v_mu;
    let x_mu2 = &x_mu * &x_mu;
    let pre_exp = -&(&x_mu2 / &v_sigma2);
    let gaussian = &v_scale * &pre_exp.apply("exp", MyTensor::exp, MyTensor::exp);
    let sample_y = RcTerm::new("y", zeros.clone());
    let diff = &gaussian - &sample_y;
    let loss = (&diff * &diff).apply_t(Box::new(Summer));

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
