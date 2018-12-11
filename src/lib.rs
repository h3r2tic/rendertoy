mod backend;
mod blob;
mod consts;
mod mesh;
mod package;
mod shader;
mod texture;

pub use gl;
pub use snoozy::*;

pub use self::blob::*;
pub use self::consts::*;
pub use self::mesh::*;
pub use self::shader::*;
pub use self::texture::*;

#[macro_use]
extern crate failure;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate lazy_static;

use glutin::dpi::*;
use glutin::GlContext;

extern "system" fn gl_debug_message(
    _source: u32,
    type_: u32,
    id: u32,
    severity: u32,
    _len: i32,
    message: *const i8,
    _param: *mut std::ffi::c_void,
) {
    unsafe {
        let s = std::ffi::CStr::from_ptr(message);

        let is_ignored_id = id == 131216; // Program/shader state info: GLSL shader * failed to compile. WAT.

        if gl::DEBUG_TYPE_PERFORMANCE == type_
            || gl::DEBUG_SEVERITY_NOTIFICATION == severity
            || is_ignored_id
        {
            println!("GL debug: {}\n", s.to_string_lossy());
        } else {
            panic!("OpenGL Debug message ({}): {}", id, s.to_string_lossy());
        }
    }
}

pub struct Rendertoy {
    events_loop: glutin::EventsLoop,
    gl_window: glutin::GlWindow,
}

impl Rendertoy {
    pub fn new() -> Rendertoy {
        let events_loop = glutin::EventsLoop::new();
        let window = glutin::WindowBuilder::new()
            .with_title("Hello, world!")
            .with_dimensions(LogicalSize::new(1024.0, 768.0));
        let context = glutin::ContextBuilder::new()
            .with_vsync(true)
            .with_gl_debug_flag(true)
            .with_gl_profile(glutin::GlProfile::Core)
            .with_gl(glutin::GlRequest::Specific(glutin::Api::OpenGl, (4, 3)));
        let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();

        unsafe {
            gl_window.make_current().unwrap();
        }

        gl::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _);

        unsafe {
            gl::DebugMessageCallback(gl_debug_message, std::ptr::null_mut());
            gl::DebugMessageControl(
                gl::DONT_CARE,
                gl::DONT_CARE,
                gl::DONT_CARE,
                0,
                std::ptr::null_mut(),
                1,
            );
            gl::DebugMessageControl(
                gl::DEBUG_SOURCE_SHADER_COMPILER,
                gl::DONT_CARE,
                gl::DONT_CARE,
                0,
                std::ptr::null_mut(),
                0,
            );

            gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl::Enable(gl::FRAMEBUFFER_SRGB);

            let mut vao: u32 = 0;
            gl::GenVertexArrays(1, &mut vao);
            gl::BindVertexArray(vao);
        }

        Rendertoy {
            events_loop,
            gl_window,
        }
    }

    fn next_frame(&mut self) -> bool {
        self.gl_window.swap_buffers().unwrap();

        let mut running = true;

        let mut events = Vec::new();
        self.events_loop.poll_events(|event| events.push(event));

        for event in events.iter() {
            #[allow(clippy::single_match)]
            match event {
                glutin::Event::WindowEvent { event, .. } => match event {
                    glutin::WindowEvent::CloseRequested => running = false,
                    glutin::WindowEvent::Resized(logical_size) => {
                        let dpi_factor = self.gl_window.get_hidpi_factor();
                        let phys_size = logical_size.to_physical(dpi_factor);
                        self.gl_window.resize(phys_size);
                        unsafe {
                            gl::Viewport(0, 0, phys_size.width as i32, phys_size.height as i32);
                        }
                    }
                    _ => (),
                },
                _ => (),
            }
        }

        running
    }

    pub fn with_frame_snapshot<F>(&mut self, callback: &mut F) -> bool
    where
        F: FnMut(&mut Snapshot),
    {
        with_snapshot(|snapshot| {
            unsafe {
                //gl::ClearColor(t.sin().abs(), 1.0, 0.1, 0.0);
                gl::Clear(gl::COLOR_BUFFER_BIT);
            }

            callback(snapshot);

            self.next_frame()
        })
    }

    pub fn forever<F>(&mut self, mut callback: F)
    where
        F: FnMut(&mut Snapshot),
    {
        let mut running = true;
        while running {
            running = self.with_frame_snapshot(&mut callback);
        }
    }
}

pub fn draw_fullscreen_texture(tex: &Texture) {
    backend::draw::draw_fullscreen_texture(tex.texture_id);
}
