use rustograd::{Tape, TapeTerm};
use std::fs::File;
use std::io::{BufWriter, Write};

#[derive(Clone, Copy)]
struct Vec2<T> {
    x: T,
    y: T,
}

impl<T: Copy> Vec2<T> {
    fn map<U>(&self, f: impl Fn(T) -> U) -> Vec2<U> {
        Vec2 {
            x: f(self.x),
            y: f(self.y),
        }
    }
}

impl<T: std::ops::Add<Output = T>> std::ops::Add for Vec2<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: std::ops::Sub<Output = T>> std::ops::Sub for Vec2<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T: std::ops::Mul<Output = T> + Copy> std::ops::Mul<T> for Vec2<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<T: std::ops::Div<Output = T> + Copy> std::ops::Div<T> for Vec2<T> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl<T: std::ops::Neg<Output = T>> std::ops::Neg for Vec2<T> {
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
        model.missile_hist.len(),
        model.target_hist.len()
    );

    let mut f = File::create("missile_model.csv")
        .map(BufWriter::new)
        .unwrap();
    writeln!(&mut f, "t, x1, y1, vx1, vy1, x2, y2, vx2, vy2").unwrap();

    let mut loss_f = File::create("missile_loss.csv")
        .map(BufWriter::new)
        .unwrap();

    let mut traj_f = File::create("missile_traj.csv")
        .map(BufWriter::new)
        .unwrap();
    writeln!(traj_f, "t, missile_x, missile_y, target_x, target_y").unwrap();
    let mut rng = Xor128::new(12321);
    for t in 0..30 {
        let (thrust, heading) = optimize(&model, t, &mut f, &mut loss_f);
        // tape.dump_nodes();
        let (pos, target) = simulate_step(&model, &mut rng, t, heading, thrust);
        writeln!(
            traj_f,
            "{t}, {}, {}, {}, {}, {thrust}, {heading}",
            pos.x, pos.y, target.x, target.y
        )
        .unwrap();
    }
}

const MAX_THRUST: f64 = 0.11;
const RATE: f64 = 3e-4;
const GM: f64 = 0.06;
const DRAG: f64 = 0.05;
const TARGET_X: f64 = 20.;
const TARGET_VX: f64 = 0.5;

fn optimize(model: &Model, t: usize, f: &mut impl Write, loss_f: &mut impl Write) -> (f64, f64) {
    for (t2, target) in model.target_hist.iter().enumerate() {
        target
            .x
            .set(TARGET_X - TARGET_VX * (t + t2) as f64)
            .unwrap();
        target.y.set(5.).unwrap();
        target.x.eval();
        target.y.eval();
    }

    let mut d_thrust = 0.;
    let mut d_heading = 0.;
    let mut thrust = 0.;
    let mut heading = 0.;

    for _ in 0..30 {
        model.loss.eval();
        model.loss.backprop().unwrap();
        d_thrust = model.thrust.grad().unwrap();
        d_heading = model.heading.grad().unwrap();
        thrust = (model.thrust.data().unwrap() - d_thrust * RATE)
            .min(MAX_THRUST)
            .max(0.);
        model.thrust.set(thrust).unwrap();
        heading = model.heading.data().unwrap() - d_heading * RATE;
        model.heading.set(heading).unwrap();
    }

    print_csv(&model, f);
    let loss_val = model.loss.eval_noclear();
    println!(
        "thrust: {}, heading: {}, loss: {}",
        d_thrust, d_heading, loss_val
    );
    writeln!(loss_f, "{}, {}", t, loss_val).unwrap();

    (thrust, heading)
}

fn simulate_step(
    model: &Model,
    rng: &mut Xor128,
    t: usize,
    heading: f64,
    thrust: f64,
) -> (Vec2<f64>, Vec2<f64>) {
    let thrust_vec = Vec2::<f64> {
        x: heading.sin() * thrust,
        y: heading.cos() * thrust,
    };
    let missile = model.missile_hist.first().unwrap();
    let velo = missile.velo.map(|x| x.data().unwrap());
    let velolen2 = velo.x * velo.x + velo.y * velo.y;
    let velolen12 = velolen2.sqrt();
    let randomize = Vec2 {
        x: rng.next() - 0.5,
        y: rng.next() - 0.5,
    };
    let accel =
        Vec2::<f64> { x: 0., y: -GM } + randomize * 0.02 - velo * DRAG / velolen12 + thrust_vec;
    let delta_x2 = velo + accel * 0.5;
    let oldpos = missile.pos.map(|x| x.data().unwrap());
    let newpos = oldpos + delta_x2;
    missile.pos.x.set(newpos.x).unwrap();
    missile.pos.y.set(newpos.y).unwrap();
    let newvelo = missile.velo.map(|x| x.data().unwrap()) + accel;
    missile.velo.x.set(newvelo.x).unwrap();
    missile.velo.y.set(newvelo.y).unwrap();
    for (t2, target) in model.target_hist.iter().enumerate() {
        target
            .x
            .set(TARGET_X - TARGET_VX * (t + t2) as f64)
            .unwrap();
    }
    let target_pos = model.target_hist.first().unwrap().map(|x| x.eval());
    (oldpos, target_pos)
}

fn print_csv(model: &Model, writer: &mut impl Write) {
    for (t, (b1, b2)) in model
        .missile_hist
        .iter()
        .zip(model.target_hist.iter())
        .enumerate()
    {
        writeln!(
            writer,
            "{}, {}, {}, {}, {}, {}, {}",
            t,
            b1.pos.x.data().unwrap(),
            b1.pos.y.data().unwrap(),
            b1.velo.x.eval_noclear(),
            b1.velo.y.eval_noclear(),
            b2.x.data().unwrap(),
            b2.y.data().unwrap()
        )
        .unwrap();
    }
}

#[derive(Clone, Copy)]
struct Missile<'a> {
    pos: Vec2<TapeTerm<'a>>,
    velo: Vec2<TapeTerm<'a>>,
    accel: Vec2<TapeTerm<'a>>,
}

impl<'a> Missile<'a> {
    fn simulate_model(
        &mut self,
        heading: TapeTerm<'a>,
        thrust: TapeTerm<'a>,
        c: &Constants<'a>,
        hist: &mut Vec<Missile<'a>>,
    ) {
        let thrust_vec = Vec2 {
            x: heading.apply("sin", |x| x.sin(), |x| x.cos()) * thrust,
            y: heading.apply("cos", |x| x.cos(), |x| -x.sin()) * thrust,
        };
        let velolen2 = self.velo.x * self.velo.x + self.velo.y * self.velo.y;
        let velolen12 = velolen2.apply("sqrt", |x| x.sqrt(), |x| 1. / 2. * x.powf(-1. / 2.));
        self.accel = gravity(c.zero, c.gm) - self.velo * c.drag / velolen12 + thrust_vec;
        let delta_x2 = self.velo + self.accel * c.half;
        self.pos = self.pos + delta_x2;
        self.velo = self.velo + self.accel;
        hist.push(*self);
    }
}

/// Even constants need nodes in autograd
struct Constants<'a> {
    gm: TapeTerm<'a>,
    zero: TapeTerm<'a>,
    half: TapeTerm<'a>,
    drag: TapeTerm<'a>,
}

/// A model for the simulation state
struct Model<'a> {
    missile_hist: Vec<Missile<'a>>,
    target_hist: Vec<Vec2<TapeTerm<'a>>>,
    heading: TapeTerm<'a>,
    thrust: TapeTerm<'a>,
    loss: TapeTerm<'a>,
}

fn get_model<'a>(tape: &'a Tape<f64>) -> Model<'a> {
    let heading = tape.term("heading", std::f64::consts::PI / 4.);
    let thrust = tape.term("thrust", 0.01);
    let missile = Missile {
        pos: Vec2 {
            x: tape.term("x1", 0.),
            y: tape.term("y1", 0.),
        },
        velo: Vec2 {
            x: tape.term("vx1", 0.05),
            y: tape.term("vy1", 0.05),
        },
        accel: Vec2 {
            x: tape.term("ax1", 0.),
            y: tape.term("ay1", 0.),
        },
    };

    let constants = Constants {
        gm: tape.term("GM", GM),
        zero: tape.term("0.0", 0.0),
        half: tape.term("0.5", 0.5),
        drag: tape.term("drag", DRAG),
    };

    let mut missile1 = missile;
    let mut hist1 = vec![missile1];
    let mut hist2 = vec![];

    hist2.push(Vec2 {
        x: tape.term("x2", 11.),
        y: tape.term("x2", 5.),
    });
    for t in 0..20 {
        missile1.simulate_model(heading, thrust, &constants, &mut hist1);
        hist2.push(Vec2 {
            x: tape.term("x2", TARGET_X - TARGET_VX * (t as f64)),
            y: tape.term("x2", 5.),
        });
    }

    let loss = hist1
        .iter()
        .zip(hist2.iter())
        .fold(None, |acc: Option<TapeTerm<'a>>, cur| {
            let diff = *cur.1 - cur.0.pos;
            let loss = diff.x * diff.x + diff.y * diff.y;
            if let Some(acc) = acc {
                Some(acc.apply_bin(loss, Box::new(MinOp)))
            } else {
                Some(loss)
            }
        })
        .unwrap();

    Model {
        missile_hist: hist1,
        target_hist: hist2,
        thrust,
        heading,
        loss,
    }
}

fn gravity<'a>(zero: TapeTerm<'a>, gm: TapeTerm<'a>) -> Vec2<TapeTerm<'a>> {
    Vec2 { x: zero, y: -gm }
}

pub(crate) struct Xor128 {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

impl Xor128 {
    pub fn new(seed: u32) -> Self {
        let mut ret = Xor128 {
            x: 294742812,
            y: 3863451937,
            z: 2255883528,
            w: 824091511,
        };
        if 0 < seed {
            ret.x ^= seed;
            ret.y ^= ret.x;
            ret.z ^= ret.y;
            ret.w ^= ret.z;
            ret.nexti();
        }
        ret.nexti();
        ret
    }

    pub fn nexti(&mut self) -> u32 {
        // T = (I + L^a)(I + R^b)(I + L^c)
        // a = 13, b = 17, c = 5
        let t = self.x ^ (self.x << 15);
        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        self.w ^= (self.w >> 21) ^ (t ^ (t >> 4));
        self.w
    }

    pub fn next(&mut self) -> f64 {
        self.nexti() as f64 / 0xffffffffu32 as f64
    }
}
