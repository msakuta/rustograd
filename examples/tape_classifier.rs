//! Least squares fitting to a Gaussian distribution using gradient descent.

use rustograd::{Tape, TapeTerm};

use std::io::Write;

fn main() {
    let tape = Tape::new();
    let model = build_model(&tape);

    let samples: Vec<_> = vec![
        [3., -2.4, 1.],
        [2.1, -1.5, 1.],
        [3.5, -2.0, 1.],
        [4.5, -0.5, 1.],
        [-3., 2.4, 0.],
        [-2.1, 1.5, 0.],
        [-3.5, 2.0, 0.],
        [-3.2, 0.5, 0.],
    ];

    let calc_loss = || {
        samples
            .iter()
            .map(|&[x, y, label]| {
                model.x.set(x).unwrap();
                model.y.set(y).unwrap();
                model.label.set(label).unwrap();
                model.loss.eval()
            })
            .sum::<f64>()
    };

    const RATE: f64 = 0.1;
    const INIT_OFFSET: f64 = 0.;
    const INIT_W0: f64 = 1.;
    const INIT_W1: f64 = 1.;

    let optimize = |offset: &mut f64, w0: &mut f64, w1: &mut f64| {
        model.offset.set(*offset).unwrap();
        model.w0.set(*w0).unwrap();
        model.w1.set(*w1).unwrap();
        for &[x, y, label] in samples.iter() {
            model.x.set(x).unwrap();
            model.y.set(y).unwrap();
            model.label.set(label).unwrap();
            model.loss.eval();
            model.loss.backprop().unwrap();
            *offset -= RATE * model.offset.grad().unwrap();
            *w0 -= RATE * model.w0.grad().unwrap();
            *w1 -= RATE * model.w1.grad().unwrap();
        }
    };

    let mut offset_val = INIT_OFFSET;
    let mut w0_val = INIT_W0;
    let mut w1_val = INIT_W1;
    let mut history = vec![];
    for i in 0..5000 {
        let t = i as f64;
        if history
            .last()
            .map(|last: &(f64, _, _, _)| last.0 * 1.1 < t)
            .unwrap_or(true)
        {
            history.push((t, offset_val, w0_val, w1_val));
        }
        println!(
            "i: {i}, offset: {offset_val}, w0: {w0_val}, w1: {w1_val}, loss: {}",
            calc_loss()
        );
        optimize(&mut offset_val, &mut w0_val, &mut w1_val);
    }

    let mut file = std::io::BufWriter::new(std::fs::File::create("classify.csv").unwrap());
    writeln!(file, "offset, w0, w1").unwrap();
    for &(t, offset, w0, w1) in history.iter() {
        // let xval = i as f64 / 20. * std::f64::consts::PI;
        writeln!(file, "{t}, {offset}, {w0}, {w1}").unwrap();
    }

    // let counter = std::cell::Cell::new(0);
    // let callback = |nodes: &_, idx| {
    //     let i = counter.get();
    //     let mut file =
    //         std::io::BufWriter::new(std::fs::File::create(format!("dot{i}.dot")).unwrap());
    //     model
    //         .loss
    //         .dot_builder()
    //         .show_values(false)
    //         .vertical(true)
    //         .highlights(idx)
    //         .dot_borrowed(nodes, &mut file)
    //         .unwrap();
    //     counter.set(i + 1);
    // };

    // model.loss.eval_cb(&callback);
    // model.loss.backprop_cb(&callback).unwrap();
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
    y: TapeTerm<'a>,
    offset: TapeTerm<'a>,
    w0: TapeTerm<'a>,
    w1: TapeTerm<'a>,
    sigmoid: TapeTerm<'a>,
    label: TapeTerm<'a>,
    loss: TapeTerm<'a>,
}

fn build_model(tape: &Tape) -> Model {
    let x = tape.term("x", 0.);
    let y = tape.term("y", 0.);
    let label = tape.term("label", 0.);
    let offset = tape.term("offset", 0.);
    let w0 = tape.term("w0", 0.);
    let w1 = tape.term("w1", 1.);
    let arg = x * w0 + y * w1 + offset;
    let one = tape.term("1", 1.);
    let sigmoid = one / (one + (-arg).apply("exp", f64::exp, f64::exp));
    let diff = sigmoid - label;
    let loss = diff * diff;
    Model {
        x,
        y,
        offset,
        w0,
        w1,
        sigmoid,
        label,
        loss,
    }
}
