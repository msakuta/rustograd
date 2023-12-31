use std::io::Write;

use rustograd::{Tape, TapeTerm};

#[derive(Clone, Copy)]
struct Vec2<'a> {
    x: TapeTerm<'a>,
    y: TapeTerm<'a>,
}

impl<'a> std::ops::Add for Vec2<'a> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<'a> std::ops::Sub for Vec2<'a> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<'a> std::ops::Mul<TapeTerm<'a>> for Vec2<'a> {
    type Output = Self;
    fn mul(self, rhs: TapeTerm<'a>) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<'a> std::ops::Div<TapeTerm<'a>> for Vec2<'a> {
    type Output = Self;
    fn div(self, rhs: TapeTerm<'a>) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl<'a> std::ops::Neg for Vec2<'a> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

fn main() {
    let tape = Tape::new();
    let model = get_model(&tape);

    let last_accel = model.accels.last().unwrap();
    let last_pos = model.xs.last().unwrap();

    // if let Ok(f) = std::fs::File::create("graph.dot") {
    //     let x2 = model.xs.first().unwrap();
    //     x2.x.dot_builder()
    //         .vertical(true)
    //         .output_term(x2.x, "x2.x")
    //         .output_term(x2.y, "x2.y")
    //         .dot(&mut std::io::BufWriter::new(f))
    //         .unwrap();
    // }

    println!("size: {}, xs: {}", tape.len(), model.xs.len());

    let mut f = std::fs::File::create("orbit.csv")
        .map(std::io::BufWriter::new)
        .unwrap();
    writeln!(&mut f, "t, x, y, vx, vy").unwrap();
    print_csv(&model, &mut f);

    let v0 = model.vs.first().unwrap();

    model.loss.eval();
    model.loss.backprop().unwrap();
    let xd = v0.x.grad().unwrap();
    let yd = v0.y.grad().unwrap();

    let eval = [last_pos.x.eval(), last_pos.y.eval()];
    println!(
        "eval: {eval:?}, accel.eval: {:?}",
        [last_accel.x.eval(), last_accel.y.eval()]
    );
    println!("derive(vx, vy): {:?}, {:?}", xd, yd);

    const RATE: f64 = 5e-5;

    let mut loss_f = std::fs::File::create("orbit_loss.csv")
        .map(std::io::BufWriter::new)
        .unwrap();

    // optimization loop
    for i in 0..30 {
        model.loss.eval();
        model.loss.backprop().unwrap();
        let xd = v0.x.grad().unwrap();
        let yd = v0.y.grad().unwrap();
        let first_velo = model.vs.first().unwrap();
        first_velo
            .x
            .set(first_velo.x.data().unwrap() - xd * RATE)
            .unwrap();
        first_velo
            .y
            .set(first_velo.y.data().unwrap() - yd * RATE)
            .unwrap();
        let loss_val = model.loss.eval();
        println!("derive(vx, vy): {:?}, {:?}, loss: {}", xd, yd, loss_val);

        print_csv(&model, &mut f);
        writeln!(&mut loss_f, "{}, {}", i, loss_val).unwrap();
    }
}

fn print_csv(model: &Model, writer: &mut impl Write) {
    for (t, (xy, v)) in model.xs.iter().zip(model.vs.iter()).enumerate() {
        writeln!(
            writer,
            "{}, {}, {}, {}, {}",
            t,
            xy.x.eval(),
            xy.y.eval(),
            v.x.eval(),
            v.y.eval()
        )
        .unwrap();
    }
}

const GM: f64 = 0.03;

struct Model<'a> {
    accels: Vec<Vec2<'a>>,
    vs: Vec<Vec2<'a>>,
    xs: Vec<Vec2<'a>>,
    loss: TapeTerm<'a>,
}

fn get_model<'a>(tape: &'a Tape<f64>) -> Model<'a> {
    let mut pos = Vec2 {
        x: tape.term("x", 1.),
        y: tape.term("y", 0.),
    };
    let mut vx = Vec2 {
        x: tape.term("vx", 0.),
        y: tape.term("vy", 0.15),
    };
    let earth = Vec2 {
        x: tape.term("bx", 0.),
        y: tape.term("by", 0.),
    };

    let gm = tape.term("GM", GM);

    let half = tape.term("0.5", 0.5);
    let mut accels = vec![];
    let mut vs = vec![vx];
    let mut xs = vec![pos];
    for _ in 0..30 {
        let accel = gravity(earth, pos, gm);
        let delta_x = vx + accel * half;
        let accel2 = gravity(earth, pos + delta_x * half, gm);
        let delta_x2 = vx + accel * half;
        pos = pos + delta_x2;
        accels.push(accel2);
        xs.push(pos);
        vx = vx + accel2;
        vs.push(vx);
    }

    let target = Vec2 {
        x: tape.term("target_x", -1.),
        y: tape.term("target_y", 0.),
    };
    let last_pos = xs[18];
    let diff = last_pos - target;
    let loss = diff.x * diff.x + diff.y * diff.y;

    Model {
        accels,
        xs,
        vs,
        loss,
    }
}

fn gravity<'a>(earth: Vec2<'a>, pos: Vec2<'a>, gm: TapeTerm<'a>) -> Vec2<'a> {
    let diff = earth - pos;
    let len2 = diff.x * diff.x + diff.y * diff.y;
    let len32 = len2.apply(
        "pow[3/2]",
        |x| x.powf(3. / 2.),
        |x| 3. / 2. * x.powf(1. / 2.),
    );
    diff / len32 * gm
}
