use std::{cell::RefCell, collections::HashMap};

use eframe::{
    egui::{self, Color32, Frame, Pos2, Rect, Ui},
    emath::Align2,
    epaint::{pos2, vec2, FontId},
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
    next_term: Option<TapeTerm<'a, f64>>,
    tick_count: usize,
    nodes: Vec<Node>,
}

impl<'a> RustographApp<'a> {
    pub fn new() -> Self {
        let tape = Box::leak(Box::new(Tape::new()));
        let a = tape.term("a", 1.23);
        let exp_a = (-a * a).apply_t(Box::new(ExpFn));
        Self {
            tape,
            a,
            next_term: Some(exp_a),
            tick_count: 0,
            nodes: vec![],
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

            for node in self.nodes.iter() {
                let rect = node.rect();
                // let center = pos2(x, NODE_OFFSET + 1 as f32 * NODE_INTERVAL);
                painter.rect_stroke(to_screen.transform_rect(rect), 10., (1., Color32::BLACK));

                let mid = rect.min.x + rect.width() / 2.;
                for parent in &node.parents {
                    if let Some(parent) = self.nodes.get(*parent) {
                        let prect = parent.rect();
                        let pmid = prect.min.x + prect.width() / 2.;
                        painter.line_segment(
                            [
                                to_screen.transform_pos(pos2(mid, rect.min.y)),
                                to_screen.transform_pos(pos2(pmid, prect.max.y)),
                            ],
                            (2., Color32::BLACK),
                        );
                        painter.circle_stroke(
                            to_screen.transform_pos(pos2(pmid, prect.max.y)),
                            4.,
                            (1., Color32::BLACK),
                        );
                    }
                }

                painter.text(
                    to_screen.transform_pos(rect.center()),
                    Align2::CENTER_CENTER,
                    &node.name,
                    FontId::monospace(12.),
                    Color32::BLACK,
                );
            }
        });
    }

    fn gen_nodes(&mut self) {
        let out_nodes = RefCell::new(vec![]);

        let gen_count = RefCell::new(vec![]);
        let gen_pos = RefCell::new(HashMap::new());
        let callback = |nodes: &[TapeNode<f64>], _idx, _| {
            for (i, node) in nodes.iter().enumerate() {
                let gen = ancestory_generations(&nodes, i);
                let mut gen_count = gen_count.borrow_mut();
                if gen_count.len() <= gen {
                    gen_count.resize(gen + 1, 0);
                }
                gen_count[gen] += 1;
                let sibling = gen_count[gen];
                gen_pos.borrow_mut().insert(i, (gen, sibling));
                let pos = node_pos(sibling, gen);

                let parents = node
                    .parents()
                    .into_iter()
                    .filter_map(|p| p.map(|p| p as usize))
                    .collect();

                let name = node.name();
                out_nodes.borrow_mut().push(Node {
                    name: name.to_string(),
                    pos,
                    width: name.len() as f32 * 10.,
                    parents,
                });
            }
        };

        if let Some(term) = self.next_term {
            self.next_term = term.gen_graph_cb(&self.a, &callback, false);
        }

        self.nodes = out_nodes.into_inner();
    }
}

const GEN_INTERVAL: usize = 100;

impl<'a> eframe::App for RustographApp<'a> {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        if self.tick_count < GEN_INTERVAL * 3 {
            if self.tick_count % GEN_INTERVAL == 0 {
                self.gen_nodes();
            }
        }
        self.tick_count += 1;

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

fn node_pos(x: usize, y: usize) -> Pos2 {
    pos2(
        NODE_OFFSET + x as f32 * NODE_X_INTERVAL,
        NODE_OFFSET + y as f32 * NODE_Y_INTERVAL,
    )
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

struct Node {
    name: String,
    pos: Pos2,
    width: f32,
    parents: Vec<usize>,
}

impl Node {
    fn rect(&self) -> Rect {
        Rect {
            min: self.pos,
            max: self.pos + vec2(self.width, NODE_HEIGHT),
        }
    }
}
