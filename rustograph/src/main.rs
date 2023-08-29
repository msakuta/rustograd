mod app;
mod exp;

use rustograd::Tape;

use crate::app::RustographApp;

fn main() {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "DeepRender",
        native_options,
        Box::new(|_cc| Box::new(RustographApp::new())),
    )
    .unwrap();
}
