use std::collections::HashMap;

use eframe::{
    egui::{self, Color32, Frame, Pos2, Rect, Ui},
    emath::Align2,
    epaint::{pos2, FontId},
    glow::ATOMIC_COUNTER_BUFFER_REFERENCED_BY_FRAGMENT_SHADER,
};
use rustograd::{tape::TapeNode, Tape, TapeTerm};

use crate::exp::ExpFn;

const NODE_OFFSET: f32 = 20.;
const NODE_WIDTH: f32 = 90.;
const NODE_HEIGHT: f32 = 30.;
const NODE_X_INTERVAL: f32 = 120.;
const NODE_Y_INTERVAL: f32 = 50.;

pub struct RustographApp<'a> {
    tape: &'static Tape<f64>,
    a: TapeTerm<'a, f64>,
    exp_a: TapeTerm<'a, f64>,
    count: usize,
}

impl<'a> RustographApp<'a> {
    pub fn new() -> Self {
        let tape = Box::leak(Box::new(Tape::new()));
        let a = tape.term("a", 1.23);
        let exp_a = (-a * a).apply_t(Box::new(ExpFn));
        Self {
            tape,
            a,
            exp_a,
            count: 0,
        }
    }

    fn paint_graph(&mut self, ui: &mut Ui) {
        Frame::canvas(ui.style()).show(ui, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::hover());

            let to_screen = egui::emath::RectTransform::from_to(
                Rect::from_min_size(Pos2::ZERO, response.rect.size()),
                response.rect,
            );

            let nodes = self.tape.nodes();
            let mut gen_count = vec![];
            let mut gen_pos = HashMap::new();

            for (i, node) in nodes.iter().enumerate() {
                let gen = ancestory_generations(&nodes, i);
                if gen_count.len() <= gen {
                    gen_count.resize(gen + 1, 0);
                }
                gen_count[gen] += 1;
                let sibling = gen_count[gen];
                gen_pos.insert(i, (gen, sibling));
                let bbox = bbox_pos(sibling, gen);
                let rect = Rect {
                    min: pos2(bbox[0], bbox[1]),
                    max: pos2(bbox[2], bbox[3]),
                };
                // let center = pos2(x, NODE_OFFSET + 1 as f32 * NODE_INTERVAL);
                painter.rect_stroke(to_screen.transform_rect(rect), 10., (1., Color32::BLACK));
                painter.text(
                    pos2(rect.min.x, rect.min.y + rect.height()),
                    Align2::LEFT_CENTER,
                    node.name(),
                    FontId::monospace(12.),
                    Color32::BLACK,
                );
            }
        });
    }
}

const GEN_INTERVAL: usize = 100;

impl<'a> eframe::App for RustographApp<'a> {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        if self.count < GEN_INTERVAL * 3 {
            if self.count % GEN_INTERVAL == 0 {
                self.exp_a.gen_graph(&self.a);
            }
        }
        self.count += 1;

        // eframe::egui::SidePanel::right("side_panel")
        //     .min_width(200.)
        //     .show(ctx, |ui| self.ui_panel(ui));

        egui::CentralPanel::default()
            // .resizable(true)
            // .min_height(100.)
            .show(ctx, |ui| {
                self.paint_graph(ui);
            });
    }
}

fn bbox_pos(x: usize, y: usize) -> [f32; 4] {
    [
        NODE_OFFSET + x as f32 * NODE_X_INTERVAL,
        NODE_OFFSET + y as f32 * NODE_Y_INTERVAL,
        NODE_OFFSET + x as f32 * NODE_X_INTERVAL + NODE_WIDTH,
        NODE_OFFSET + y as f32 * NODE_Y_INTERVAL + NODE_HEIGHT,
    ]
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
