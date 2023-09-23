use eframe::{
    egui::{self, Color32, Frame, Pos2, Rect, Ui},
    epaint::{pos2, vec2},
};
use rustograd::{Tape, TapeTerm};

use crate::lander::{lander_simulate_step, simulate_lander, LanderModel, LanderState, Vec2};

const SCALE: f32 = 10.;

pub struct LanderApp<'a> {
    tape: &'static Tape<f64>,
    a: TapeTerm<'a, f64>,
    t: f32,
    playback_speed: f32,
    paused: bool,
    direct_control: bool,
    lander_state: LanderState,
    lander_model: LanderModel,
    h_thrust: f64,
    v_thrust: f64,
    error_msg: Option<String>,
}

const LANDER_STATE: LanderState = LanderState {
    pos: Vec2 { x: 2., y: 10. },
    velo: Vec2 { x: 0., y: 0. },
    heading: 0.,
    prediction: vec![],
};

impl<'a> LanderApp<'a> {
    pub fn new() -> Self {
        let tape = Box::leak(Box::new(Tape::new()));
        let a = tape.term("a", 1.23);
        let lander_state = LANDER_STATE;
        let lander_model = simulate_lander(Vec2 { x: 2., y: 10. }).unwrap();
        Self {
            tape,
            a,
            t: 0.,
            playback_speed: 0.1,
            paused: false,
            direct_control: false,
            lander_state,
            lander_model,
            h_thrust: 0.,
            v_thrust: 0.,
            error_msg: None,
        }
    }

    fn paint_graph(&mut self, ui: &mut Ui) {
        Frame::canvas(ui.style()).show(ui, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::click());

            let to_screen = egui::emath::RectTransform::from_to(
                Rect::from_min_size(Pos2::ZERO, response.rect.size()),
                response.rect,
            );
            let from_screen = to_screen.inverse();

            let to_pos2 = |pos: Vec2<f64>| {
                to_screen.transform_pos(pos2(
                    (pos.x as f32 + 20.) * SCALE,
                    (20. - pos.y as f32) * SCALE,
                ))
            };

            let to_vec2 = |pos: Vec2<f64>| vec2(pos.x as f32 * SCALE, -pos.y as f32 * SCALE);

            let from_pos2 = |pos: Pos2| {
                let model_pos = from_screen.transform_pos(pos);
                Vec2 {
                    x: (model_pos.x / SCALE - 20.) as f64,
                    y: (20. - model_pos.y / SCALE) as f64,
                }
            };

            if response.clicked() {
                if let Some(mouse_pos) = response.interact_pointer_pos() {
                    match simulate_lander(from_pos2(mouse_pos)) {
                        Ok(res) => self.lander_model = res,
                        Err(e) => self.error_msg = Some(e.to_string()),
                    }
                    self.t = 0.;
                    println!("New lander_model");
                }
            }

            painter.circle(
                to_pos2(self.lander_model.target),
                5.,
                Color32::RED,
                (1., Color32::BROWN),
            );

            let render_lander = |lander_state: &LanderState| {
                let pos = lander_state
                    .prediction
                    .iter()
                    .map(|x| to_pos2(*x))
                    .collect();
                let path =
                    eframe::epaint::PathShape::line(pos, (2., Color32::from_rgb(127, 127, 0)));
                painter.add(path);

                painter.circle(
                    to_pos2(lander_state.pos),
                    5.,
                    Color32::BLUE,
                    (1., Color32::RED),
                );

                painter.arrow(
                    to_pos2(lander_state.pos),
                    to_vec2(lander_state.velo * 2.),
                    (2., Color32::from_rgb(127, 0, 127)).into(),
                );

                painter.arrow(
                    to_pos2(lander_state.pos),
                    to_vec2(
                        Vec2 {
                            x: lander_state.heading.sin(),
                            y: -lander_state.heading.cos(),
                        } * 2.,
                    ),
                    (2., Color32::BLUE).into(),
                );
            };

            if self.direct_control {
                render_lander(&self.lander_state);
            } else if let Some(lander_state) = self.lander_model.lander_states.get(self.t as usize)
            {
                render_lander(lander_state);
            } else {
                self.t = 0.;
            }
        });
    }
}

impl<'a> eframe::App for LanderApp<'a> {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        eframe::egui::SidePanel::right("side_panel")
            .min_width(200.)
            .show(ctx, |ui| {
                ui.checkbox(&mut self.direct_control, "direct_control");
                if ui.button("Reset").clicked() {
                    self.lander_state = LANDER_STATE;
                    self.t = 0.;
                }
                ui.checkbox(&mut self.paused, "Paused");
                ui.label("Playback speed (times 60fps):");
                ui.add(egui::widgets::Slider::new(
                    &mut self.playback_speed,
                    (0.1)..=2.,
                ));
            });

        egui::CentralPanel::default()
            // .resizable(true)
            // .min_height(100.)
            .show(ctx, |ui| {
                self.paint_graph(ui);
            });

        if self.direct_control {
            ctx.input(|input| {
                self.h_thrust = 0.;
                self.v_thrust = 0.;
                for key in input.keys_down.iter() {
                    match key {
                        egui::Key::A => self.h_thrust = -1.,
                        egui::Key::D => self.h_thrust = 1.,
                        egui::Key::W => self.v_thrust = 1.,
                        _ => {}
                    }
                }
            });

            lander_simulate_step(
                &mut self.lander_state,
                self.h_thrust,
                self.v_thrust,
                self.playback_speed as f64,
            );
        }

        if !self.paused {
            self.t += self.playback_speed;
        }
    }
}
