//! Peak separation using least squares fitting with gradient descent.

use rustograd::{Tape, TapeTerm};

use std::io::Write;

fn main() {
    let tape = Tape::new();
    let model = build_model(&tape);

    fn truth(x: f64) -> f64 {
        0.75 * (-(x - 0.8).powf(2.) / 0.35).exp() + 0.5 * (-(x + 0.3).powf(2.) / 0.5).exp()
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
    const INIT_MU0: f64 = -1.;
    const INIT_MU1: f64 = 1.;
    const INIT_SIGMA: f64 = 1.;
    const INIT_SCALE: f64 = 1.;

    let optimize = |mu0: &mut f64,
                    sigma0: &mut f64,
                    scale0: &mut f64,
                    mu1: &mut f64,
                    sigma1: &mut f64,
                    scale1: &mut f64| {
        model.mu0.set(*mu0).unwrap();
        model.sigma0.set(*sigma0).unwrap();
        model.scale0.set(*scale0).unwrap();
        model.mu1.set(*mu1).unwrap();
        model.sigma1.set(*sigma1).unwrap();
        model.scale1.set(*scale1).unwrap();
        for (&xval, &sample_y) in samples.iter().zip(truth_data.iter()) {
            model.x.set(xval).unwrap();
            model.sample_y.set(sample_y).unwrap();
            model.loss.eval();
            model.loss.backprop().unwrap();
            *mu0 -= RATE * model.mu0.grad().unwrap();
            *sigma0 -= RATE * model.sigma0.grad().unwrap();
            *scale0 -= RATE * model.scale0.grad().unwrap();
            *mu1 -= RATE * model.mu1.grad().unwrap();
            *sigma1 -= RATE * model.sigma1.grad().unwrap();
            *scale1 -= RATE * model.scale1.grad().unwrap();
        }
    };

    let mut mu0 = INIT_MU0;
    let mut sigma0 = INIT_SIGMA;
    let mut scale0 = INIT_SCALE;
    let mut mu1 = INIT_MU1;
    let mut sigma1 = INIT_SIGMA;
    let mut scale1 = INIT_SCALE;
    let mut history: Vec<[f64; 7]> = vec![];
    for i in 0..500 {
        optimize(
            &mut mu0,
            &mut sigma0,
            &mut scale0,
            &mut mu1,
            &mut sigma1,
            &mut scale1,
        );
        let t = i as f64;
        if history
            .last()
            .map(|&[last, ..]| last * 1.2 < t)
            .unwrap_or(true)
        {
            history.push([t, mu0, sigma0, scale0, mu1, sigma1, scale1]);
        }
        println!(
            "i: {i}, mu0: {mu0}, sigma0: {sigma0}, scale0: {scale0}, mu1: {mu1}, sigma1: {sigma1}, scale1: {scale1}, loss: {}",
            calc_loss()
        );
    }

    let mut file = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    writeln!(file, "x, y, init_y, truth_y").unwrap();
    for (&xval, &truth_y) in samples.iter().zip(truth_data.iter()) {
        // let xval = i as f64 / 20. * std::f64::consts::PI;
        model.x.set(xval).unwrap();
        model.mu0.set(mu0).unwrap();
        model.sigma0.set(sigma0).unwrap();
        model.scale0.set(scale0).unwrap();
        model.mu1.set(mu1).unwrap();
        model.sigma1.set(sigma1).unwrap();
        model.scale1.set(scale1).unwrap();
        let value = model.y.eval();
        model.mu0.set(INIT_MU0).unwrap();
        model.sigma0.set(INIT_SIGMA).unwrap();
        model.scale0.set(INIT_SCALE).unwrap();
        model.mu1.set(INIT_MU1).unwrap();
        model.sigma1.set(INIT_SIGMA).unwrap();
        model.scale1.set(INIT_SCALE).unwrap();
        let init_value = model.y.eval();
        let hist_string = history
            .iter()
            .map(|&[_, mu0, sigma0, scale0, mu1, sigma1, scale1]| {
                model.mu0.set(mu0).unwrap();
                model.sigma0.set(sigma0).unwrap();
                model.scale0.set(scale0).unwrap();
                model.mu1.set(mu1).unwrap();
                model.sigma1.set(sigma1).unwrap();
                model.scale1.set(scale1).unwrap();
                (model.g0.eval(), model.g1.eval(), model.y.eval())
            })
            .fold("".to_string(), |acc, (g0, g1, y)| {
                acc + &format!(", {g0}, {g1}, {y}")
            });
        writeln!(
            file,
            "{xval}, {value}, {init_value}, {truth_y}{hist_string}"
        )
        .unwrap();
    }

    let counter = std::cell::Cell::new(0);
    let callback = |nodes: &_, idx| {
        let i = counter.get();
        let mut file =
            std::io::BufWriter::new(std::fs::File::create(format!("dot{i}.dot")).unwrap());
        model
            .loss
            .dot_builder()
            .show_values(false)
            .vertical(true)
            .highlights(idx)
            .dot_borrowed(nodes, &mut file)
            .unwrap();
        counter.set(i + 1);
    };

    model.x.set(0.).unwrap();
    model.loss.eval_cb(&callback);
    model.loss.backprop_cb(&callback).unwrap();
    let mut dotfile = std::io::BufWriter::new(std::fs::File::create("graph.dot").unwrap());
    model
        .loss
        .dot_builder()
        .vertical(true)
        .dot(&mut dotfile)
        .unwrap();
}

struct Model<'a> {
    x: TapeTerm<'a>,
    mu0: TapeTerm<'a>,
    sigma0: TapeTerm<'a>,
    scale0: TapeTerm<'a>,
    mu1: TapeTerm<'a>,
    g0: TapeTerm<'a>,
    sigma1: TapeTerm<'a>,
    scale1: TapeTerm<'a>,
    sample_y: TapeTerm<'a>,
    g1: TapeTerm<'a>,
    y: TapeTerm<'a>,
    loss: TapeTerm<'a>,
}

fn build_model(tape: &Tape) -> Model {
    let x = tape.term("x", 0.);

    let gaussian = |i: i32| {
        let mu = tape.term(format!("mu{i}"), 0.);
        let sigma = tape.term(format!("sigma{i}"), 1.);
        let scale = tape.term(format!("scale{i}"), 1.);
        let x_mu = x - mu;
        let g = scale * (-(x_mu * x_mu) / sigma / sigma).apply("exp", f64::exp, f64::exp);
        (mu, sigma, scale, g)
    };

    let (mu0, sigma0, scale0, g0) = gaussian(0);
    let (mu1, sigma1, scale1, g1) = gaussian(1);
    let y = g0 + g1;

    let sample_y = tape.term("y", 0.);
    let diff = y - sample_y;
    let loss = diff * diff;
    Model {
        x,
        mu0,
        sigma0,
        scale0,
        mu1,
        g0,
        sigma1,
        scale1,
        g1,
        y,
        sample_y,
        loss,
    }
}
