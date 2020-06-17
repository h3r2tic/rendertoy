#![feature(core_intrinsics)]
#![feature(async_closure)]

pub extern crate tracing;

#[macro_use]
extern crate failure;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate snoozy_macros;
#[macro_use]
extern crate abomonation_derive;

mod backend;
mod blob;
mod buffer;
mod camera;
mod consts;
mod dot;
mod gpu_debugger;
mod gpu_profiler;
mod gui;
mod keyboard;
mod math;
mod mesh;
mod package;
mod renderer;
mod rendertoy;
mod rgb9e5;
mod shader;
mod texture;
mod viewport;
mod vk_backend_state;
mod vk_render_device;
mod vulkan;
mod warnings;

pub mod compute_tex_macro;

pub use self::blob::*;
pub use self::buffer::*;
pub use self::camera::*;
pub use self::consts::*;
pub use self::keyboard::*;
pub use self::mesh::*;
pub use self::rendertoy::*;
pub use self::rgb9e5::*;
pub use self::shader::*;
pub use self::texture::*;
pub use self::viewport::*;
pub use ash::{vk, vk::Format};
pub use math::*;
pub use snoozy::*;
pub use warnings::rtoy_show_warning;

#[global_allocator]
static ALLOC: rpmalloc::RpMalloc = rpmalloc::RpMalloc;

pub trait RenderPass: Any {
    fn prepare_frame(
        &mut self,
        view_constants: &ViewConstants,
        frame_state: &FrameState,
        frame_idx: u32,
    );
}

use std::any::Any;
pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
}

impl<T> AsAny for T
where
    T: RenderPass,
{
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Blanket-implement a trait combining RenderPass and AsAny
// for anything that implements RenderPass + AsAny.
pub trait RenderPassAny: RenderPass + AsAny {}
impl<T> RenderPassAny for T where T: RenderPass + AsAny {}

impl<F> RenderPass for F
where
    F: FnMut(&ViewConstants, &FrameState, u32),
    F: 'static,
{
    fn prepare_frame(
        &mut self,
        view_constants: &ViewConstants,
        frame_state: &FrameState,
        frame_idx: u32,
    ) {
        (*self)(view_constants, frame_state, frame_idx)
    }
}

pub type RenderPassList = Vec<Box<dyn RenderPassAny>>;

// Convenient .add method which adds a trait object to the render pass list,
// but returns a borrow of the concrete type just added. Supports the following pattern:
//
// let ao_tex = sub_passes
//     .add(rtoy_samples::ssao::Ssao::new(tex_key, gbuffer_tex.clone()))
//     .get_output_tex();
pub trait AddRenderPass {
    fn add<P: RenderPassAny + 'static>(&mut self, pass: P) -> &P;
}

impl AddRenderPass for RenderPassList {
    fn add<P: RenderPassAny + 'static>(&mut self, pass: P) -> &P {
        self.push(Box::new(pass));
        self.last().unwrap().as_any().downcast_ref::<P>().unwrap()
    }
}
