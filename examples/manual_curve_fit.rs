//! Least squares fitting to a Gaussian distribution using gradient descent with hand calculated gradients.
//! Provided to benchmark the performance of autograd.

use std::io::Write;

fn main() {
    fn truth(x: f64) -> f64 {
        0.75 * (-(x - 1.2).powf(2.) / 0.35).exp()
    }

    fn model_fn(x: f64, mu: f64, sigma: f64, scale: f64) -> f64 {
        scale * (-(x - mu).powf(2.) / sigma / sigma).exp()
    }

    let samples: Vec<_> = (-40..40).map(|i| i as f64 / 10.).collect();
    let truth_data: Vec<_> = samples.iter().map(|x| truth(*x)).collect();

    let calc_loss = |mu, sigma, scale| {
        samples
            .iter()
            .map(|&xval| model_fn(xval, mu, sigma, scale))
            .sum::<f64>()
    };

    const RATE: f64 = 0.01;
    const INIT_MU: f64 = 0.;
    const INIT_SIGMA: f64 = 1.;
    const INIT_SCALE: f64 = 1.;

    let optimize = |mu: &mut f64, sigma: &mut f64, scale: &mut f64| {
        for (&xval, &sample_y) in samples.iter().zip(truth_data.iter()) {
            let predicted_y = model_fn(xval, *mu, *sigma, *scale);
            let dmu =
                -2. * (sample_y - predicted_y) * 2. * (xval - *mu) / *sigma / *sigma * predicted_y;
            let dsigma =
                -2. * (sample_y - predicted_y) * 2. * (xval - *mu).powf(2.) / *sigma / *sigma
                    * predicted_y;
            let dscale =
                -2. * (sample_y - predicted_y) * (-(xval - *mu).powf(2.) / *sigma / *sigma).exp();
            *mu -= RATE * dmu;
            *sigma -= RATE * dsigma;
            *scale -= RATE * dscale;
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
            calc_loss(mu_val, sigma_val, scale_val)
        );
    }

    let mut file = std::io::BufWriter::new(std::fs::File::create("data.csv").unwrap());
    writeln!(file, "x, y, init_y, truth_y").unwrap();
    for (&xval, &truth_y) in samples.iter().zip(truth_data.iter()) {
        // let xval = i as f64 / 20. * std::f64::consts::PI;
        let value = model_fn(xval, mu_val, sigma_val, scale_val);
        let init_value = model_fn(xval, INIT_MU, INIT_SIGMA, INIT_SCALE);
        let hist_string = history
            .iter()
            .map(|&(_, mu_val, sigma_val, scale_val)| model_fn(xval, mu_val, sigma_val, scale_val))
            .fold("".to_string(), |acc, cur| acc + &format!(", {cur}"));
        writeln!(
            file,
            "{xval}, {value}, {init_value}, {truth_y}{hist_string}"
        )
        .unwrap();
    }
}
