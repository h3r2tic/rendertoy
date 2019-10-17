use std::collections::HashMap;
use std::default::Default;
use std::mem::replace;
use std::sync::Mutex;

pub fn report_texture(name: &str, texture_handle: u32) {
    GPU_DEBUGGER
        .lock()
        .unwrap()
        .report_texture(name, texture_handle);
}

pub fn with_textures<F: FnOnce(&GpuDebuggerTextures)>(f: F) {
    f(&GPU_DEBUGGER.lock().unwrap().textures);
}

pub fn end_frame() {
    GPU_DEBUGGER.lock().unwrap().clear();
}

#[derive(Default, Debug)]
pub struct GpuDebuggerTextures {
    pub textures: HashMap<String, u32>,
}

struct GpuDebugger {
    textures: GpuDebuggerTextures,
}

impl GpuDebugger {
    pub fn new() -> Self {
        Self {
            textures: Default::default(),
        }
    }

    fn clear(&mut self) {
        self.textures.textures.clear();
    }

    fn report_texture(&mut self, name: &str, texture_handle: u32) {
        self.textures
            .textures
            .insert(name.to_string(), texture_handle);
    }
}

lazy_static! {
    static ref GPU_DEBUGGER: Mutex<GpuDebugger> = { Mutex::new(GpuDebugger::new()) };
}
