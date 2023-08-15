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
        [-1.2, 1.5, 0.],
        [-0.6, 2.4, 2.],
        [0.1, 3.2, 2.],
        [0.3, 2.9, 2.],
        [-0.2, 2.5, 2.],
    ];

    let calc_loss = || {
        samples
            .iter()
            .map(|&[x, y, label]| {
                model.x.set(x).unwrap();
                model.y.set(y).unwrap();
                for (i, w) in model.ws.iter().enumerate() {
                    w.one_hot.set((label as usize == i) as i32 as f64).unwrap();
                }
                model.loss.eval()
            })
            .sum::<f64>()
    };

    const RATE: f64 = 0.01;
    const INIT_OFFSET: f64 = 0.;
    const INIT_W0: f64 = 0.25;
    const INIT_W1: f64 = 0.25;
    const INIT_W2: f64 = -0.125;
    const INIT_W3: f64 = -0.175;
    const INIT_W4: f64 = 0.325;
    const INIT_W5: f64 = -0.2175;

    let optimize = |ws: &mut [Weights<f64>]| {
        for (model, w) in model.ws.iter().zip(ws.iter()) {
            model.offset.set(w.offset).unwrap();
            model.w0.set(w.w0).unwrap();
            model.w1.set(w.w1).unwrap();
        }
        for &[x, y, label] in samples.iter() {
            model.x.set(x).unwrap();
            model.y.set(y).unwrap();
            for (i, w) in model.ws.iter().enumerate() {
                w.one_hot.set((label as usize == i) as i32 as f64).unwrap();
            }
            model.loss.eval();
            model.loss.backprop().unwrap();
            for (model, w) in model.ws.iter().zip(ws.iter_mut()) {
                w.offset -= RATE * model.offset.grad().unwrap();
                w.w0 -= RATE * model.w0.grad().unwrap();
                w.w1 -= RATE * model.w1.grad().unwrap();
            }
        }
    };

    const WS: [Weights<f64>; 3] = [
        Weights {
            offset: INIT_OFFSET,
            w0: INIT_W0,
            w1: INIT_W1,
            one_hot: 0.,
        },
        Weights {
            offset: INIT_OFFSET,
            w0: INIT_W2,
            w1: INIT_W3,
            one_hot: 0.,
        },
        Weights {
            offset: INIT_OFFSET,
            w0: INIT_W4,
            w1: INIT_W5,
            one_hot: 0.,
        },
    ];
    let ws_to_string = |w: &Weights<f64>| format!("{}, {}, {}", w.offset, w.w0, w.w1);
    let mut ws = WS;
    let mut history = vec![];
    for i in 0..200 {
        let t = i as f64;
        if history
            .last()
            .map(|last: &(f64, _)| last.0 * 1.1 < t)
            .unwrap_or(true)
        {
            history.push((t, ws));
        }
        println!(
            "i: {i}, w0: {}, w1: {}, w2: {} loss: {}",
            ws_to_string(&ws[0]),
            ws_to_string(&ws[1]),
            ws_to_string(&ws[2]),
            calc_loss()
        );
        optimize(&mut ws);
    }

    let mut file = std::io::BufWriter::new(std::fs::File::create("classify.csv").unwrap());
    writeln!(
        file,
        "offset0, w00, w01, offset1, w10, w11, offset2, w20, w21"
    )
    .unwrap();
    for &(t, ws) in history.iter() {
        writeln!(
            file,
            "{t}, {}, {}, {}",
            ws_to_string(&ws[0]),
            ws_to_string(&ws[1]),
            ws_to_string(&ws[2])
        )
        .unwrap();
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

#[derive(Clone, Copy)]
struct Weights<T> {
    w0: T,
    w1: T,
    offset: T,
    one_hot: T,
}

struct Model<'a> {
    x: TapeTerm<'a>,
    y: TapeTerm<'a>,
    ws: [Weights<TapeTerm<'a>>; 3],
    loss: TapeTerm<'a>,
}

fn build_model(tape: &Tape) -> Model {
    let x = tape.term("x", 0.);
    let y = tape.term("y", 0.);
    let new_weights = |i| Weights::<_> {
        w0: tape.term(format!("w{i}_0"), 0.),
        w1: tape.term(format!("w{i}_1"), 1.),
        offset: tape.term(format!("offset{i}"), 0.),
        one_hot: tape.term(format!("one_hot{i}"), 0.),
    };
    let w0 = new_weights(0);
    let w1 = new_weights(1);
    let w2 = new_weights(2);
    let arg1 = (x * w0.w0 + y * w0.w1 + w0.offset).apply("exp", f64::exp, f64::exp);
    let arg2 = (x * w1.w0 + y * w1.w1 + w1.offset).apply("exp", f64::exp, f64::exp);
    let arg3 = (x * w2.w0 + y * w2.w1 + w2.offset).apply("exp", f64::exp, f64::exp);
    let sum = arg1 + arg2 + arg3;
    let softmax1 = arg1 / sum;
    let softmax2 = arg2 / sum;
    let softmax3 = arg3 / sum;
    let softmaxes = [softmax1, softmax2, softmax3];
    let log_softmax = softmaxes.iter().map(|x| x.apply("ln", f64::ln, |x| 1. / x));
    // let weights_reg = |w: &Weights<_>| w.w0 * w.w0 + w.w1 + w.w1;
    // let regularizer = weights_reg(&w0) + weights_reg(&w1) + weights_reg(&w2);
    // let regular_factor = tape.term("lambda", 1.);
    let loss = -[&w0, &w1, &w2]
        .iter()
        .zip(log_softmax)
        .map(|(w, log_sm)| w.one_hot * log_sm)
        .reduce(|acc, cur| acc + cur)
        .unwrap();
    // + regular_factor * regularizer;
    Model {
        x,
        y,
        ws: [w0, w1, w2],
        loss,
    }
}
