use rustograd::{Tape, TapeTerm};
use std::io::Write;

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

struct MinOp;

impl rustograd::BinaryFn<f64> for MinOp {
    fn name(&self) -> String {
        "min".to_string()
    }

    fn f(&self, lhs: f64, rhs: f64) -> f64 {
        lhs.min(rhs)
    }

    fn t(&self, data: f64) -> (f64, f64) {
        (data, data)
    }

    fn grad(&self, lhs: f64, rhs: f64) -> (f64, f64) {
        if lhs < rhs {
            (1., 0.)
        } else {
            (0., 1.)
        }
    }
}

fn main() {
    let tape = Tape::new();
    let model = get_model(&tape);

    let first1 = model.hist1.first().unwrap();
    let first2 = model.hist2.first().unwrap();

    // if let Ok(f) = std::fs::File::create("graph.dot") {
    //     let x2 = model.hist1.first().unwrap().pos;
    //     x2.x.dot_builder()
    //         .vertical(true)
    //         .output_term(x2.x, "x2.x")
    //         .output_term(x2.y, "x2.y")
    //         .dot(&mut std::io::BufWriter::new(f))
    //         .unwrap();
    // }

    println!(
        "size: {}, b1: {}, b2: {}",
        tape.len(),
        model.hist1.len(),
        model.hist2.len()
    );

    let mut f = std::fs::File::create("cannon2.csv")
        .map(std::io::BufWriter::new)
        .unwrap();
    writeln!(&mut f, "t, x1, y1, vx1, vy1, x2, y2, vx2, vy2").unwrap();
    print_csv(&model, &mut f);

    let v1 = first1.velo;
    let v2 = first2.velo;

    model.loss.eval();
    model.loss.backprop().unwrap();
    // tape.dump_nodes();

    let xd1 = v1.x.grad().unwrap();
    let yd1 = v1.y.grad().unwrap();
    let xd2 = v2.x.grad().unwrap();
    let yd2 = v2.y.grad().unwrap();

    let eval = [first1.pos.x.eval(), first1.pos.y.eval()];
    println!(
        "eval: {eval:?}, accel.eval: {:?}",
        [first1.accel.x.eval(), first1.accel.y.eval()]
    );
    println!("derive(vx, vy): {:?}, {:?}, {:?}, {:?}", xd1, yd1, xd2, yd2);

    const RATE: f64 = 3e-4;

    let mut loss_f = std::fs::File::create("cannon_loss.csv")
        .map(std::io::BufWriter::new)
        .unwrap();

    // optimization loop
    for i in 0..30 {
        model.loss.eval();
        model.loss.backprop().unwrap();
        let xd1 = v1.x.grad().unwrap();
        let yd1 = v1.y.grad().unwrap();
        v1.x.set(v1.x.data().unwrap() - xd1 * RATE).unwrap();
        v1.y.set(v1.y.data().unwrap() - yd1 * RATE).unwrap();
        let xd2 = v2.x.grad().unwrap();
        let yd2 = v2.y.grad().unwrap();
        v2.x.set(v2.x.data().unwrap() - xd2 * RATE).unwrap();
        v2.y.set(v2.y.data().unwrap() - yd2 * RATE).unwrap();
        let loss_val = model.loss.eval();
        println!(
            "derive(v1): {}, {}, derive(v2): {}, {}, loss: {}",
            xd1, yd1, xd2, xd2, loss_val
        );

        print_csv(&model, &mut f);
        writeln!(&mut loss_f, "{}, {}", i, loss_val).unwrap();
    }
}

fn print_csv(model: &Model, writer: &mut impl Write) {
    for (t, (b1, b2)) in model.hist1.iter().zip(model.hist2.iter()).enumerate() {
        writeln!(
            writer,
            "{}, {}, {}, {}, {}, {}, {}, {}, {}",
            t,
            b1.pos.x.eval(),
            b1.pos.y.eval(),
            b1.velo.x.eval(),
            b1.velo.y.eval(),
            b2.pos.x.eval(),
            b2.pos.y.eval(),
            b2.velo.x.eval(),
            b2.velo.y.eval()
        )
        .unwrap();
    }
}

const GM: f64 = 0.03;

#[derive(Clone, Copy)]
struct Bullet<'a> {
    pos: Vec2<'a>,
    velo: Vec2<'a>,
    accel: Vec2<'a>,
}

struct Model<'a> {
    hist1: Vec<Bullet<'a>>,
    hist2: Vec<Bullet<'a>>,
    loss: TapeTerm<'a>,
}

fn get_model<'a>(tape: &'a Tape<f64>) -> Model<'a> {
    let mut bullet1 = Bullet {
        pos: Vec2 {
            x: tape.term("x1", 0.),
            y: tape.term("y1", 0.),
        },
        velo: Vec2 {
            x: tape.term("vx1", 0.5),
            y: tape.term("vy1", 0.5),
        },
        accel: Vec2 {
            x: tape.term("ax1", 0.),
            y: tape.term("ay1", 0.),
        },
    };
    let mut bullet2 = Bullet {
        pos: Vec2 {
            x: tape.term("x2", 10.),
            y: tape.term("y2", 2.),
        },
        velo: Vec2 {
            x: tape.term("vx2", -0.5),
            y: tape.term("vy2", 0.5),
        },
        accel: Vec2 {
            x: tape.term("ax2", 0.),
            y: tape.term("ay2", 0.),
        },
    };

    let gm = tape.term("GM", GM);

    let zero = tape.term("0.0", 0.0);
    let half = tape.term("0.5", 0.5);
    let drag = tape.term("drag", 0.02);
    let mut hist1 = vec![bullet1];
    let mut hist2 = vec![bullet2];
    let simulate_bullet = |bullet: &mut Bullet<'a>, hist: &mut Vec<Bullet<'a>>| {
        let velolen2 = bullet.velo.x * bullet.velo.x + bullet.velo.y * bullet.velo.y;
        let velolen12 = velolen2.apply("sqrt", |x| x.sqrt(), |x| 1. / 2. * x.powf(-1. / 2.));
        bullet.accel = gravity(zero, gm) - bullet.velo * drag / velolen12;
        let delta_x2 = bullet.velo + bullet.accel * half;
        bullet.pos = bullet.pos + delta_x2;
        bullet.velo = bullet.velo + bullet.accel;
        hist.push(*bullet);
    };

    for _ in 0..20 {
        simulate_bullet(&mut bullet1, &mut hist1);
        simulate_bullet(&mut bullet2, &mut hist2);
    }

    let loss = hist1
        .iter()
        .zip(hist2.iter())
        .fold(None, |acc: Option<TapeTerm<'a>>, cur| {
            let diff = cur.1.pos - cur.0.pos;
            let loss = diff.x * diff.x + diff.y * diff.y;
            if let Some(acc) = acc {
                Some(acc.apply_bin(loss, Box::new(MinOp)))
            } else {
                Some(loss)
            }
        })
        .unwrap();

    Model { hist1, hist2, loss }
}

fn gravity<'a>(zero: TapeTerm<'a>, gm: TapeTerm<'a>) -> Vec2<'a> {
    Vec2 { x: zero, y: -gm }
}
