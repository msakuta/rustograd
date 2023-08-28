use gif::{Encoder, Frame, Repeat};
use rustograd::tape::{add_mul, TapeIndex, TapeNode};
use rustograd::{Tape, UnaryFn};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;

struct ExpFn;

impl UnaryFn<f64> for ExpFn {
    fn name(&self) -> String {
        "exp".to_string()
    }

    fn f(&self, data: f64) -> f64 {
        data.exp()
    }

    fn grad(&self, data: f64) -> f64 {
        data.exp()
    }

    fn gen_graph(
        &self,
        nodes: &mut Vec<TapeNode<f64>>,
        _input: TapeIndex,
        output: TapeIndex,
        derived: TapeIndex,
        optim: bool,
    ) -> Option<TapeIndex> {
        Some(add_mul(nodes, output, derived, optim))
    }
}

fn main() {
    let tape = Tape::new();
    let a = tape.term("a", 1.23);
    let exp_a = (-a * a).apply_t(Box::new(ExpFn));

    let color_map = &[0xFF, 0xFF, 0xFF, 0, 0, 0];
    let (width, height) = (640, 480);
    // let frames = 10;
    let (width_u, height_u) = (width as usize, height as usize);
    let mut image = File::create("target/beacon.gif").unwrap();
    let mut encoder = Encoder::new(&mut image, width, height, color_map).unwrap();
    encoder.set_repeat(Repeat::Infinite).unwrap();
    let encoder = std::cell::RefCell::new(encoder);
    let callback = |nodes: &[TapeNode<f64>], idx, _| {
        let mut buffer = vec![0u8; (width_u * height_u) as usize];
        let mut frame = Frame::default();
        let mut gen_count = vec![];
        let mut gen_pos = HashMap::new();
        println!("inserting {} to {}", idx, nodes.len());
        for (i, node) in nodes.iter().enumerate() {
            let gen = ancestory_generations(nodes, i);
            if gen_count.len() <= gen {
                gen_count.resize(gen + 1, 0);
            }
            gen_count[gen] += 1;
            let sibling = gen_count[gen];
            gen_pos.insert(i, (gen, sibling));
            let bbox = bbox_pos(sibling, gen);
            rectangle(&mut buffer, (width_u, height_u), bbox);
            let mid = (bbox[0] + bbox[2]) / 2;
            for parent in nodes[i].parents().into_iter().filter_map(|p| p) {
                if let Some(parent) = gen_pos.get(&(parent as usize)) {
                    let pbbox = bbox_pos(parent.1, parent.0);
                    let pmid = (pbbox[0] + pbbox[2]) / 2;
                    line(
                        &mut buffer,
                        (width_u, height_u),
                        [mid, bbox[1]],
                        [pmid, pbbox[3]],
                    );
                    rectangle(
                        &mut buffer,
                        (width_u, height_u),
                        [pmid - 4, pbbox[3] - 4, pmid + 4, pbbox[3] + 4],
                    );
                }
            }
            for j in 0..i {
                rectangle(
                    &mut buffer,
                    (width_u, height_u),
                    [
                        30 + sibling * 90 + 10 + j % 10 * 5,
                        30 + gen * 50 + 10 + j / 10 * 5,
                        30 + sibling * 90 + 10 + j % 10 * 5 + 4,
                        30 + gen * 50 + 10 + j / 10 * 5 + 4,
                    ],
                );
            }
        }
        frame.delay = 50;
        frame.width = width;
        frame.height = height;
        frame.buffer = Cow::Borrowed(&buffer);
        encoder.borrow_mut().write_frame(&frame).unwrap();
    };

    let next = std::cell::Cell::new(exp_a);
    let mut derivatives = vec![exp_a];
    // write!(csv, "x, $e^x$, ").unwrap();
    for _i in 1..3 {
        let next_node = next.get().gen_graph_cb(&a, &callback, false).unwrap();
        derivatives.push(next_node);
        // write!(csv, "$(d^{i} \\exp(-x^2)/(d x^{i})$, ").unwrap();
        next.set(next_node);
    }
}

fn bbox_pos(x: usize, y: usize) -> [usize; 4] {
    [30 + x * 90, 30 + y * 50, 90 + x * 90, 60 + y * 50]
}

fn clamp(v: usize, min: usize, max: usize) -> usize {
    v.max(min).min(max)
}

fn rectangle(buffer: &mut [u8], (width, height): (usize, usize), [l, t, r, b]: [usize; 4]) {
    for y in clamp(t, 0, height)..clamp(b, 0, height) {
        if l < width {
            buffer[l + y * width] = 1;
        }
        if r < width {
            buffer[r + y * width] = 1;
        }
    }
    for x in clamp(l, 0, width)..=clamp(r, 0, width - 1) {
        if t < height {
            buffer[x + t * width] = 1;
        }
        if b < height {
            buffer[x + b * width] = 1;
        }
    }
}

fn line(
    buffer: &mut [u8],
    (width, height): (usize, usize),
    [mut x0, mut y0]: [usize; 2],
    [mut x1, mut y1]: [usize; 2],
) {
    let mut polarity = false;
    if x1 < x0 {
        std::mem::swap(&mut x0, &mut x1);
        polarity = !polarity;
    }
    if y1 < y0 {
        std::mem::swap(&mut y0, &mut y1);
        polarity = !polarity;
    }
    let pixels = (x1 - x0).max(y1 - y0);
    for i in 0..pixels {
        let x = (x1 - x0) * i / pixels + x0;
        let y = if polarity {
            y1 - (y1 - y0) * i / pixels
        } else {
            (y1 - y0) * i / pixels + y0
        };
        if x < width && y < width {
            buffer[x + y * width] = 1;
        }
    }
}

fn ancestory_generations(nodes: &[TapeNode<f64>], idx: usize) -> usize {
    nodes[idx]
        .parents()
        .into_iter()
        .filter_map(|p| p)
        .map(|p| ancestory_generations(nodes, p as usize) + 1)
        .max()
        .unwrap_or(0)
}
