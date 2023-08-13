//! Least squares fitting to a Gaussian distribution using gradient descent.

use rustograd::{Tape, TapeTerm};

use std::io::Write;

const INITIAL_POSITIONS: [[f64; 2]; 7] = [
    [0., 0.],
    [0.5, 0.],
    [1., 0.],
    [1.5, 0.5],
    [2., 1.],
    [2., 1.5],
    [2., 2.],
];

fn main() {
    let tape = Tape::new();
    let model = build_model(&tape);

    let calc_loss = || model.loss.eval();

    const RATE: f64 = 0.1;

    let optimize = |xs: &mut [f64], ys: &mut [f64]| {
        for (xval, xterm) in xs.iter().zip(model.x.iter()) {
            xterm.set(*xval).unwrap();
        }
        for (yval, yterm) in ys.iter().zip(model.y.iter()) {
            yterm.set(*yval).unwrap();
        }
        model.loss.eval();
        model.loss.backprop().unwrap();
        let len = xs.len() - 2;
        for (xval, xterm) in xs.iter_mut().zip(model.x.iter()).skip(1).take(len) {
            *xval -= RATE * xterm.grad().unwrap();
        }
        for (yval, yterm) in ys.iter_mut().zip(model.y.iter()).skip(1).take(len) {
            *yval -= RATE * yterm.grad().unwrap();
        }
    };

    let mut xs: Vec<_> = INITIAL_POSITIONS.iter().map(|[x, _]| *x).collect();
    let mut ys: Vec<_> = INITIAL_POSITIONS.iter().map(|[_, y]| *y).collect();
    let mut history = vec![];
    for i in 0..1000 {
        optimize(&mut xs, &mut ys);
        let t = i as f64;
        if history
            .last()
            .map(|last: &(f64, _, _)| last.0 * 1.2 < t)
            .unwrap_or(true)
        {
            history.push((t, xs.clone(), ys.clone()));
        }
        println!("i: {i}, xs: {xs:?}, ys: {ys:?}, loss: {}", calc_loss());
    }

    let mut file = std::io::BufWriter::new(std::fs::File::create("path.csv").unwrap());
    writeln!(file, "x, y, init_y, truth_y").unwrap();
    for (i, ((init, &xval), &yval)) in INITIAL_POSITIONS
        .iter()
        .zip(xs.iter())
        .zip(ys.iter())
        .enumerate()
    {
        let hist_string = history
            .iter()
            .map(|(_, xs, ys)| (xs[i], ys[i]))
            .fold("".to_string(), |acc, (x, y)| acc + &format!(", {x}, {y}"));
        writeln!(
            file,
            "{}, {}, {xval}, {yval}{hist_string}",
            init[0], init[1],
        )
        .unwrap();
    }

    model.loss.eval();
    model.loss.backprop().unwrap();
    let mut dotfile = std::io::BufWriter::new(std::fs::File::create("graph.dot").unwrap());
    model
        .loss
        .dot_builder()
        .vertical(true)
        .dot(&mut dotfile)
        .unwrap();
}

type Term<'a> = TapeTerm<'a, f64>;

struct Model<'a> {
    x: Vec<Term<'a>>,
    y: Vec<Term<'a>>,
    loss: Term<'a>,
}

fn build_model(tape: &Tape) -> Model {
    let x: Vec<_> = INITIAL_POSITIONS
        .iter()
        .enumerate()
        .map(|(i, _)| tape.term(format!("x{i}"), 0.))
        .collect();
    let y: Vec<_> = INITIAL_POSITIONS
        .iter()
        .enumerate()
        .map(|(i, _)| tape.term(format!("y{i}"), 0.))
        .collect();
    let pot_x = tape.term("pot_x", 1.);
    let pot_y = tape.term("pot_y", 1.);
    let sigma2 = tape.term("4.", 4.);
    let potential = x
        .iter()
        .zip(y.iter())
        .map(|(x, y)| {
            let x_diff = *x - pot_x;
            let y_diff = *y - pot_y;
            (-(x_diff * x_diff + y_diff * y_diff) / sigma2).apply("exp", f64::exp, f64::exp)
        })
        .reduce(|acc, cur| acc + cur)
        .unwrap();
    let scale = tape.term("1.25", 1.25);
    let x_diff: Vec<_> = x
        .iter()
        .zip(x.iter().skip(1))
        .map(|(x0, x1)| *x1 - *x0)
        .collect();
    let y_diff: Vec<_> = y
        .iter()
        .zip(y.iter().skip(1))
        .map(|(x0, x1)| *x1 - *x0)
        .collect();
    let x_diff2: Vec<_> = x_diff.iter().map(|x| *x * *x).collect();
    let y_diff2: Vec<_> = y_diff.iter().map(|x| *x * *x).collect();
    let loss = x_diff2
        .iter()
        .zip(y_diff2.iter())
        .map(|(x, y)| *x + *y)
        .reduce(|acc, x| acc + x)
        .unwrap()
        + scale * potential;
    Model { x, y, loss }
}
