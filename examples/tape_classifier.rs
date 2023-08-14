//! Autograd backpropagation applied to a neural network for a classification problem.

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
                if label == 0. {
                    model.softmax1.eval()
                } else {
                    model.softmax2.eval()
                }
            })
            .sum::<f64>()
    };

    const RATE: f64 = 0.01;
    const INIT_OFFSET: f64 = 0.;
    const INIT_W0: f64 = -0.25;
    const INIT_W1: f64 = -0.25;
    const INIT_W2: f64 = 0.125;
    const INIT_W3: f64 = 0.175;

    let optimize = |offset1: &mut f64,
                    w0: &mut f64,
                    w1: &mut f64,
                    offset2: &mut f64,
                    w2: &mut f64,
                    w3: &mut f64| {
        model.offset1.set(*offset1).unwrap();
        model.w0.set(*w0).unwrap();
        model.w1.set(*w1).unwrap();
        model.offset2.set(*offset2).unwrap();
        model.w2.set(*w2).unwrap();
        model.w3.set(*w3).unwrap();
        for &[x, y, label] in samples.iter() {
            model.x.set(x).unwrap();
            model.y.set(y).unwrap();
            model.label.set(label).unwrap();
            model.loss.eval();
            model.loss.backprop().unwrap();
            *offset1 -= RATE * model.offset1.grad().unwrap();
            *w0 -= RATE * model.w0.grad().unwrap();
            *w1 -= RATE * model.w1.grad().unwrap();
            *offset2 -= RATE * model.offset2.grad().unwrap();
            *w2 -= RATE * model.w2.grad().unwrap();
            *w3 -= RATE * model.w3.grad().unwrap();
        }
    };

    let mut offset1_val = INIT_OFFSET;
    let mut w0_val = INIT_W0;
    let mut w1_val = INIT_W1;
    let mut offset2_val = INIT_OFFSET;
    let mut w2_val = INIT_W2;
    let mut w3_val = INIT_W3;
    let mut history = vec![];
    for i in 0..100 {
        let t = i as f64;
        if history
            .last()
            .map(|last: &(f64, _, _, _, _, _, _)| last.0 * 1.1 < t)
            .unwrap_or(true)
        {
            history.push((t, offset1_val, w0_val, w1_val, offset2_val, w2_val, w3_val));
        }
        println!(
            "i: {i}, offset: {offset1_val}, w0: {w0_val}, w1: {w1_val}, offset2: {offset2_val}, w0: {w2_val}, w1: {w3_val}, loss: {}",
            calc_loss()
        );
        optimize(
            &mut offset1_val,
            &mut w0_val,
            &mut w1_val,
            &mut offset2_val,
            &mut w2_val,
            &mut w3_val,
        );
    }

    let mut file = std::io::BufWriter::new(std::fs::File::create("classify.csv").unwrap());
    writeln!(file, "offset1, w0, w1, offset2, w2, w3").unwrap();
    for &(t, offset1, w0, w1, offset2, w2, w3) in history.iter() {
        writeln!(file, "{t}, {offset1}, {w0}, {w1}, {offset2}, {w2}, {w3}").unwrap();
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
        .softmax1
        .dot_builder()
        .vertical(true)
        .dot(&mut dotfile)
        .unwrap();
}

struct Model<'a> {
    x: TapeTerm<'a>,
    y: TapeTerm<'a>,
    offset1: TapeTerm<'a>,
    w0: TapeTerm<'a>,
    w1: TapeTerm<'a>,
    offset2: TapeTerm<'a>,
    w2: TapeTerm<'a>,
    w3: TapeTerm<'a>,
    label: TapeTerm<'a>,
    softmax1: TapeTerm<'a>,
    softmax2: TapeTerm<'a>,
    loss: TapeTerm<'a>,
}

fn build_model(tape: &Tape) -> Model {
    let x = tape.term("x", 0.);
    let y = tape.term("y", 0.);
    let label = tape.term("label", 0.);
    let offset1 = tape.term("offset1", 0.);
    let w0 = tape.term("w0", 0.);
    let w1 = tape.term("w1", 1.);
    let offset2 = tape.term("offset2", 0.);
    let w2 = tape.term("w2", -1.);
    let w3 = tape.term("w3", -1.);
    let arg1 = (x * w0 + y * w1 + offset1).apply("exp", f64::exp, f64::exp);
    let arg2 = (x * w2 + y * w3 + offset2).apply("exp", f64::exp, f64::exp);
    let sum = arg1 + arg2;
    let softmax1 = arg1 / sum;
    let softmax2 = arg2 / sum;
    let log_softmax1 = softmax1.apply("ln", f64::ln, |x| 1. / x);
    let log_softmax2 = softmax2.apply("ln", f64::ln, |x| 1. / x);
    let one = tape.term("1", 1.);
    let loss = label * log_softmax1 + (one - label) * log_softmax2;
    Model {
        x,
        y,
        offset1,
        w0,
        w1,
        offset2,
        w2,
        w3,
        label,
        softmax1,
        softmax2,
        loss,
    }
}
