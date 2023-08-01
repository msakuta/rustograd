//! Least squares fitting to a Gaussian distribution using gradient descent.

use rustograd::{Tape, TapeTerm};

use std::io::Write;

fn main() {
    let tape = Tape::new();
    let model = build_model(&tape);

    fn truth(x: f64) -> f64 {
        0.75 * (-(x - 1.2).powf(2.) / 0.35).exp()
    }

    let samples: Vec<_> = (-40..40).map(|i| i as f64 / 10.).collect();
    let truth_data: Vec<_> = samples.iter().map(|x| truth(*x)).collect();

    let calc_loss = || {
        samples
            .iter()
            .map(|&xval| {
                model.x.set(xval).unwrap();
                model.loss.eval()
            })
            .sum::<f64>()
    };

    const RATE: f64 = 0.01;
    const INIT_MU: f64 = 0.;
    const INIT_SIGMA: f64 = 1.;
    const INIT_SCALE: f64 = 1.;

    let optimize = |mu: &mut f64, sigma: &mut f64, scale: &mut f64| {
        model.mu.set(*mu).unwrap();
        model.sigma.set(*sigma).unwrap();
        model.scale.set(*scale).unwrap();
        for (&xval, &sample_y) in samples.iter().zip(truth_data.iter()) {
            model.x.set(xval).unwrap();
            model.sample_y.set(sample_y).unwrap();
            model.loss.eval();
            model.loss.backprop();
            *mu -= RATE * model.mu.grad();
            *sigma -= RATE * model.sigma.grad();
            *scale -= RATE * model.scale.grad();
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
            history.push((t, mu_val, sigma_val, scale_val));
        }
        println!(
            "i: {i}, mu: {mu_val}, sigma: {sigma_val}, scale: {scale_val}, loss: {}",
            calc_loss()
        );
    }

    let mut file = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    writeln!(file, "x, y, init_y, truth_y").unwrap();
    for (&xval, &truth_y) in samples.iter().zip(truth_data.iter()) {
        // let xval = i as f64 / 20. * std::f64::consts::PI;
        model.x.set(xval).unwrap();
        model.mu.set(mu_val).unwrap();
        model.sigma.set(sigma_val).unwrap();
        model.scale.set(scale_val).unwrap();
        let value = model.gaussian.eval();
        model.mu.set(INIT_MU).unwrap();
        model.sigma.set(INIT_SIGMA).unwrap();
        model.scale.set(INIT_SCALE).unwrap();
        let init_value = model.gaussian.eval();
        let hist_string = history
            .iter()
            .map(|&(_, mu_val, sigma_val, scale_val)| {
                model.mu.set(mu_val).unwrap();
                model.sigma.set(sigma_val).unwrap();
                model.scale.set(scale_val).unwrap();
                model.gaussian.eval()
            })
            .fold("".to_string(), |acc, cur| acc + &format!(", {cur}"));
        writeln!(
            file,
            "{xval}, {value}, {init_value}, {truth_y}{hist_string}"
        )
        .unwrap();
    }
    model.x.set(0.).unwrap();
    model.gaussian.eval();
    model.gaussian.backprop();
    let mut dotfile = std::io::BufWriter::new(std::fs::File::create("graph.dot").unwrap());
    model.gaussian.dot(&mut dotfile, false).unwrap();
}

struct Model<'a> {
    x: TapeTerm<'a>,
    mu: TapeTerm<'a>,
    sigma: TapeTerm<'a>,
    scale: TapeTerm<'a>,
    sample_y: TapeTerm<'a>,
    gaussian: TapeTerm<'a>,
    loss: TapeTerm<'a>,
}

fn build_model(tape: &Tape) -> Model {
    let x = tape.term("x", 0.);
    let mu = tape.term("mu", 0.);
    let sigma = tape.term("sigma", 1.);
    let scale = tape.term("scale", 1.);
    let x_mu = x - mu;
    let gaussian = scale * (-(x_mu * x_mu) / sigma / sigma).apply("exp", f64::exp, f64::exp);
    let sample_y = tape.term("y", 0.);
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
