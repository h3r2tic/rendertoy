mod backend;
mod blob;
mod buffer;
mod consts;
mod mesh;
mod package;
mod rgb9e5;
mod rt;
mod shader;
mod texture;

pub use gl;
pub use nalgebra as na;
pub use snoozy::*;

pub use self::blob::*;
pub use self::buffer::*;
pub use self::consts::*;
pub use self::mesh::*;
pub use self::rt::*;
pub use self::shader::*;
pub use self::texture::*;

pub type Point3 = na::Point3<f32>;
pub type Vector3 = na::Vector3<f32>;
pub type Matrix4 = na::Matrix4<f32>;
pub type Isometry3 = na::Isometry3<f32>;

#[macro_use]
extern crate failure;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate static_assertions;

use glutin::dpi::*;
use glutin::GlContext;

use clap::ArgMatches;

use std::str::FromStr;

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
    mouse_physical_pos: glutin::dpi::PhysicalPosition,
    last_timing_query_elapsed: u64,
    cfg: RendertoyConfig,
}

pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

pub struct FrameState {
    pub mouse_pos: Point2,
    pub gpu_time_ms: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct RendertoyConfig {
    pub width: u32,
    pub height: u32,
}

fn parse_resolution(s: &str) -> Result<(u32, u32)> {
    match s.find('x') {
        Some(pos) => match (
            FromStr::from_str(&s[..pos]),
            FromStr::from_str(&s[pos + 1..]),
        ) {
            (Ok(a), Ok(b)) => return Ok((a, b)),
            _ => (),
        },
        None => (),
    };

    Err(format_err!("Expected NUMBERxNUMBER, got {}", s))
}

impl RendertoyConfig {
    fn from_args(matches: &ArgMatches) -> RendertoyConfig {
        let (width, height) = matches
            .value_of("resolution")
            .map(|val| parse_resolution(val).unwrap())
            .unwrap_or((1280, 720));

        RendertoyConfig { width, height }
    }
}

impl Rendertoy {
    pub fn new_with_config(cfg: RendertoyConfig) -> Rendertoy {
        let events_loop = glutin::EventsLoop::new();
        let window = glutin::WindowBuilder::new()
            .with_title("Hello, rusty world!")
            .with_dimensions(LogicalSize::new(cfg.width as f64, cfg.height as f64));
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
            mouse_physical_pos: glutin::dpi::PhysicalPosition::new(0.0, 0.0),
            last_timing_query_elapsed: 0,
            cfg,
        }
    }

    pub fn new() -> Rendertoy {
        let matches = clap::App::new("Rendertoy")
            .version("1.0")
            .about("Does awesome things")
            .arg(
                clap::Arg::with_name("resolution")
                    .long("resolution")
                    .help("Window resolution")
                    .takes_value(true),
            )
            .get_matches();

        Self::new_with_config(RendertoyConfig::from_args(&matches))
    }

    pub fn width(&self) -> u32 {
        self.cfg.width
    }

    pub fn height(&self) -> u32 {
        self.cfg.height
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
                    glutin::WindowEvent::CursorMoved {
                        position: logical_pos,
                        device_id: _,
                        modifiers: _,
                    } => {
                        let dpi_factor = self.gl_window.get_hidpi_factor();
                        self.mouse_physical_pos = logical_pos.to_physical(dpi_factor);
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
        F: FnMut(&mut Snapshot, &FrameState),
    {
        with_snapshot(|snapshot| {
            unsafe {
                //gl::ClearColor(t.sin().abs(), 1.0, 0.1, 0.0);
                gl::Clear(gl::COLOR_BUFFER_BIT);
            }

            let state = FrameState {
                mouse_pos: Point2 {
                    x: self.mouse_physical_pos.x as f32,
                    y: self.mouse_physical_pos.y as f32,
                },
                gpu_time_ms: self.last_timing_query_elapsed as f64 * 1e-6,
            };

            callback(snapshot, &state);

            self.next_frame()
        })
    }

    pub fn forever<F>(&mut self, mut callback: F)
    where
        F: FnMut(&mut Snapshot, &FrameState),
    {
        let mut timing_query_handle = 0u32;
        let mut timing_query_in_flight = false;

        unsafe {
            gl::GenQueries(1, &mut timing_query_handle);
        }

        let mut running = true;
        while running {
            unsafe {
                if timing_query_in_flight {
                    let mut available: i32 = 0;
                    gl::GetQueryObjectiv(
                        timing_query_handle,
                        gl::QUERY_RESULT_AVAILABLE,
                        &mut available,
                    );

                    if available != 0 {
                        timing_query_in_flight = false;
                        gl::GetQueryObjectui64v(
                            timing_query_handle,
                            gl::QUERY_RESULT,
                            &mut self.last_timing_query_elapsed,
                        );
                    }
                }

                if !timing_query_in_flight {
                    gl::BeginQuery(gl::TIME_ELAPSED, timing_query_handle);
                }
            }

            running = self.with_frame_snapshot(&mut callback);

            unsafe {
                if !timing_query_in_flight {
                    gl::EndQuery(gl::TIME_ELAPSED);
                    timing_query_in_flight = true;
                }
            }
        }
    }
}

pub fn draw_fullscreen_texture(tex: &Texture) {
    backend::draw::draw_fullscreen_texture(tex.texture_id);
}
