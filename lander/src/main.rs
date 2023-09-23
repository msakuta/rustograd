mod app;
mod lander;

use rustograd::Tape;

use crate::app::LanderApp;

fn main() {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Lander",
        native_options,
        Box::new(|_cc| Box::new(LanderApp::new())),
    )
    .unwrap();
}
