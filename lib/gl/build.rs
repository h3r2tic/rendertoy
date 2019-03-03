extern crate gl_generator;

use gl_generator::{Api, Fallbacks, GlobalGenerator, Profile, Registry};
use std::env;
use std::fs::File;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let mut file = File::create(&Path::new(&out_dir).join("bindings.rs")).unwrap();

    Registry::new(Api::Gl, (4, 5), Profile::Core, Fallbacks::All, [
        "GL_ARB_bindless_texture"
    ])
        .write_bindings(GlobalGenerator, &mut file)
        .unwrap();
}
