use rustograd::{Tape, TapeTerm};

use std::io::Write;

#[derive(Clone, Copy, Debug)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
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

pub struct LanderState {
    pub pos: Vec2<f64>,
    pub velo: Vec2<f64>,
    pub heading: f64,
    pub prediction: Vec<Vec2<f64>>,
}

pub struct LanderModel {
    pub lander_states: Vec<LanderState>,
    pub target: Vec2<f64>,
}

pub(crate) fn simulate_lander(pos: Vec2<f64>) -> Result<LanderModel, GradDoesNotExist> {
    let tape = Tape::new();
    let model = get_model(&tape, pos);

    if let Ok(f) = std::fs::File::create("graph.dot") {
        model.loss.eval();
        model.loss.backprop().unwrap(); // To color
        let x2 = model.lander_hist.first().unwrap().pos;
        x2.x.dot_builder()
            .vertical(true)
            .output_term(x2.x, "x2.x")
            .output_term(x2.y, "x2.y")
            .output_term(model.loss, "loss")
            .dot(&mut std::io::BufWriter::new(f))
            .unwrap();
    }

    println!("size: {}, b1: {}", tape.len(), model.lander_hist.len());

    let mut rng = Xor128::new(12321);
    let lander_states = (0..100)
        .map(|t| -> Result<LanderState, GradDoesNotExist> {
            let (h_thrust, v_thrust) = optimize(&model, t)?;
            // tape.dump_nodes();
            let (pos, heading) = simulate_step(&model, &mut rng, t, h_thrust, v_thrust);
            let first = model.lander_hist.first().unwrap();
            Ok(LanderState {
                pos,
                velo: first.velo.map(|x| x.data().unwrap()),
                heading,
                prediction: model
                    .lander_hist
                    .iter()
                    .map(|b1| b1.pos.map(|x| x.data().unwrap()))
                    .collect(),
            })
        })
        .collect::<Result<Vec<_>, GradDoesNotExist>>()?;
    Ok(LanderModel {
        lander_states,
        target: model.target.map(|x| x.data().unwrap()),
    })
}

const MAX_THRUST: f64 = 0.11;
const RATE: f64 = 3e-4;
const GM: f64 = 0.06;
const DRAG: f64 = 0.05;
const TARGET_X: f64 = 20.;
const TARGET_VX: f64 = 0.5;

#[derive(Debug)]
pub struct GradDoesNotExist {
    name: String,
    file: String,
    line: u32,
}

impl GradDoesNotExist {
    fn new(name: impl Into<String>, file: impl Into<String>, line: u32) -> Self {
        Self {
            name: name.into(),
            file: file.into(),
            line,
        }
    }
}

impl std::fmt::Display for GradDoesNotExist {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Gradient does not exist in variable {} at {}:{}",
            self.name, self.file, self.line
        )
    }
}

/// A macro that attempts to get a gradient of a term, or raise GradDoesNotExist error
macro_rules! try_grad {
    ($term:expr) => {
        $term
            .grad()
            .ok_or_else(|| GradDoesNotExist::new($term.name(), file!(), line!()))?
    };
}

fn optimize(model: &Model, t: usize) -> Result<(f64, f64), GradDoesNotExist> {
    model.target.x.eval();
    model.target.y.eval();

    let mut d_h_thrust = 0.;
    let mut d_v_thrust = 0.;
    let mut v_thrust = 0.;
    let mut h_thrust = 0.;

    for _ in 0..30 {
        model.loss.eval();
        model.loss.backprop().unwrap();
        for (i, hist) in model
            .lander_hist
            .iter()
            .enumerate()
            .take(model.lander_hist.len() - 2)
        {
            d_h_thrust = try_grad!(hist.h_thrust);
            d_v_thrust = try_grad!(hist.v_thrust);
            h_thrust = (hist.h_thrust.data().unwrap() - d_h_thrust * RATE)
                .min(MAX_THRUST)
                .max(-MAX_THRUST);
            v_thrust = (hist.v_thrust.data().unwrap() - d_v_thrust * RATE)
                .min(MAX_THRUST)
                .max(0.0);
            hist.h_thrust.set(h_thrust).unwrap();
            hist.v_thrust.set(v_thrust).unwrap();
        }
    }

    let loss_val = model.loss.eval_noclear();
    println!(
        "h_thrust: {}, v_thrust: {}, d_h_thrust: {}, d_v_thrust: {}, heading: {}, loss: {}",
        h_thrust,
        v_thrust,
        d_h_thrust,
        d_v_thrust,
        model.lander_hist.first().unwrap().heading.eval_noclear(),
        loss_val
    );

    let h_thrust = model.lander_hist.first().unwrap().h_thrust.data().unwrap();
    let v_thrust = model.lander_hist.first().unwrap().v_thrust.data().unwrap();

    Ok((h_thrust, v_thrust))
}

fn simulate_step(
    model: &Model,
    rng: &mut Xor128,
    t: usize,
    h_thrust: f64,
    v_thrust: f64,
) -> (Vec2<f64>, f64) {
    let heading = model.lander_hist.first().unwrap().heading.data().unwrap();
    let thrust_vec = Vec2::<f64> {
        x: -heading.sin() * v_thrust,
        y: heading.cos() * v_thrust,
    };
    let lander = model.lander_hist.first().unwrap();
    let velo = lander.velo.map(|x| x.data().unwrap());
    let velolen2 = velo.x * velo.x + velo.y * velo.y;
    let velolen12 = velolen2.sqrt();
    let randomize = Vec2 {
        x: rng.next() - 0.5,
        y: rng.next() - 0.5,
    };
    let next_heading = heading + h_thrust;
    let accel = Vec2::<f64> { x: 0., y: -GM } + randomize * 0.02 + thrust_vec;
    let delta_x2 = velo + accel * 0.5;
    let oldpos = lander.pos.map(|x| x.data().unwrap());
    let newpos = oldpos + delta_x2;
    lander.pos.x.set(newpos.x).unwrap();
    lander.pos.y.set(newpos.y).unwrap();
    let newvelo = lander.velo.map(|x| x.data().unwrap()) + accel;
    lander.velo.x.set(newvelo.x).unwrap();
    lander.velo.y.set(newvelo.y).unwrap();
    // let target_pos = model.target.map(|x| x.eval_noclear());
    println!("Setting next_heading {next_heading}");
    lander.heading.set(next_heading).unwrap();
    (oldpos, next_heading)
}

pub fn lander_simulate_step(
    lander_state: &mut LanderState,
    // t: usize,
    h_thrust: f64,
    v_thrust: f64,
    delta_time: f64,
) {
    let heading = lander_state.heading;
    let thrust_vec = Vec2::<f64> {
        x: -heading.sin() * v_thrust,
        y: heading.cos() * v_thrust,
    };
    let velo = lander_state.velo;
    // let velolen2 = velo.x * velo.x + velo.y * velo.y;
    // let velolen12 = velolen2.sqrt();
    // let randomize = Vec2 {
    //     x: rng.next() - 0.5,
    //     y: rng.next() - 0.5,
    // };
    let next_heading = heading + h_thrust * delta_time;
    let accel = Vec2::<f64> { x: 0., y: -GM } + /*randomize * 0.02 +*/ thrust_vec;
    let delta_x2 = velo + accel * 0.5 * delta_time;
    let oldpos = lander_state.pos;
    let newpos = oldpos + delta_x2 * delta_time;
    lander_state.pos = newpos;
    let newvelo = lander_state.velo + accel * delta_time;
    lander_state.velo = newvelo;
    lander_state.heading = next_heading;
}

#[derive(Clone, Copy)]
struct LanderTape<'a> {
    h_thrust: TapeTerm<'a>,
    v_thrust: TapeTerm<'a>,
    pos: Vec2<TapeTerm<'a>>,
    velo: Vec2<TapeTerm<'a>>,
    heading: TapeTerm<'a>,
    accel: Vec2<TapeTerm<'a>>,
}

impl<'a> LanderTape<'a> {
    fn simulate_model(
        &mut self,
        tape: &'a Tape,
        c: &Constants<'a>,
        hist: &mut Vec<LanderTape<'a>>,
    ) {
        let thrust_vec = Vec2 {
            x: -self.heading.apply("sin", |x| x.sin(), |x| x.cos()) * self.v_thrust,
            y: self.heading.apply("cos", |x| x.cos(), |x| -x.sin()) * self.v_thrust,
        };
        let next_heading = self.heading + self.h_thrust;
        self.accel = gravity(c.zero, c.gm) + thrust_vec;
        let delta_x2 = self.velo + self.accel * c.half;
        self.pos = self.pos + delta_x2;
        self.velo = self.velo + self.accel;
        self.heading = next_heading;
        self.h_thrust = tape.term(format!("h_thrust{}", hist.len()), 0.);
        self.v_thrust = tape.term(format!("v_thrust{}", hist.len()), 0.);
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
    lander_hist: Vec<LanderTape<'a>>,
    target: Vec2<TapeTerm<'a>>,
    loss: TapeTerm<'a>,
}

fn get_model<'a>(tape: &'a Tape<f64>, initial_pos: Vec2<f64>) -> Model<'a> {
    let lander = LanderTape {
        h_thrust: tape.term("h_thrust", 0.01),
        v_thrust: tape.term("v_thrust", 0.01),
        pos: Vec2 {
            x: tape.term("x1", initial_pos.x),
            y: tape.term("y1", initial_pos.y),
        },
        velo: Vec2 {
            x: tape.term("vx1", 0.),
            y: tape.term("vy1", -0.01),
        },
        heading: tape.term("heading", 0.1),
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

    let mut lander1 = lander;
    let mut hist1 = vec![lander1];
    let target = Vec2 {
        x: tape.term("x2", 0.),
        y: tape.term("x2", 0.),
    };
    for t in 0..20 {
        lander1.simulate_model(tape, &constants, &mut hist1);
    }

    let loss = hist1
        .iter()
        .fold(None, |acc: Option<TapeTerm<'a>>, cur| {
            let diff = target - cur.pos;
            let loss = diff.x * diff.x + diff.y * diff.y;
            if let Some(acc) = acc {
                Some(acc.apply_bin(loss, Box::new(MinOp)))
            } else {
                Some(loss)
            }
        })
        .unwrap();

    let loss = loss
        + hist1
            .iter()
            .fold(None, |acc, cur| {
                let loss = (cur.velo.x * cur.velo.x + cur.velo.y * cur.velo.y) / (-cur.pos.y).apply("exp", f64::exp, f64::exp)
                    + cur.heading * cur.heading;
                if let Some(acc) = acc {
                    Some(acc + loss)
                } else {
                    Some(loss)
                }
            })
            .unwrap();

    Model {
        lander_hist: hist1,
        target,
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
