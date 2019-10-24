#![feature(core_intrinsics)]

mod backend;
mod blob;
mod buffer;
mod camera;
mod consts;
mod gpu_debugger;
mod gpu_profiler;
mod keyboard;
mod mesh;
mod package;
mod rgb9e5;
mod shader;
mod texture;
mod viewport;

pub use gl;
pub use nalgebra as na;
pub use snoozy::*;

pub use self::blob::*;
pub use self::buffer::*;
pub use self::camera::*;
pub use self::consts::*;
pub use self::keyboard::*;
pub use self::mesh::*;
pub use self::rgb9e5::*;
pub use self::shader::*;
pub use self::texture::*;
pub use self::viewport::*;

pub type Point2 = na::Point2<f32>;
pub type Vector2 = na::Vector2<f32>;

pub type Point3 = na::Point3<f32>;
pub type Vector3 = na::Vector3<f32>;

pub type Point4 = na::Point4<f32>;
pub type Vector4 = na::Vector4<f32>;

pub type Matrix4 = na::Matrix4<f32>;
pub type Isometry3 = na::Isometry3<f32>;

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

use clap::ArgMatches;
use glutin::dpi::*;
use glutin::GlContext;
use nanovg::{Alignment, Color, Font, TextOptions};
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

        let is_important_type = match type_ {
            gl::DEBUG_TYPE_ERROR
            | gl::DEBUG_TYPE_UNDEFINED_BEHAVIOR
            | gl::DEBUG_TYPE_DEPRECATED_BEHAVIOR
            | gl::DEBUG_TYPE_PORTABILITY => true,
            _ => false,
        };

        if !is_important_type || is_ignored_id {
            println!("GL debug: {}\n", s.to_string_lossy());
        } else {
            panic!(
                "OpenGL Debug message ({}, {:x}, {:x}): {}",
                id,
                type_,
                severity,
                s.to_string_lossy()
            );
        }
    }
}

struct GlState {
    vao: u32,
}

pub struct Rendertoy {
    events_loop: glutin::EventsLoop,
    gl_window: glutin::GlWindow,
    mouse_state: MouseState,
    cfg: RendertoyConfig,
    keyboard: KeyboardState,
    gl_state: GlState,
    selected_debug_name: Option<String>,
    last_frame_instant: std::time::Instant,
}

#[derive(Clone)]
pub struct MouseState {
    pub pos: Point2,
    pub delta: Vector2,
    pub button_mask: u32,
}

impl Default for MouseState {
    fn default() -> Self {
        Self {
            pos: Point2::origin(),
            delta: Vector2::zeros(),
            button_mask: 0,
        }
    }
}

impl MouseState {
    fn update(&mut self, new_state: &MouseState) {
        self.delta = new_state.pos - self.pos;
        self.pos = new_state.pos;
        self.button_mask = new_state.button_mask;
    }
}

pub struct FrameState<'a> {
    pub mouse: &'a MouseState,
    pub keys: &'a KeyboardState,
    pub window_size_pixels: (u32, u32),
    pub dt: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct RendertoyConfig {
    pub width: u32,
    pub height: u32,
    pub vsync: bool,
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

        let vsync = matches
            .value_of("vsync")
            .map(|val| {
                FromStr::from_str(val).expect("Could not parse the value of 'vsync' as bool")
            })
            .unwrap_or(true);

        RendertoyConfig {
            width,
            height,
            vsync,
        }
    }
}

impl Rendertoy {
    pub fn new_with_config(cfg: RendertoyConfig) -> Rendertoy {
        let events_loop = glutin::EventsLoop::new();
        let window = glutin::WindowBuilder::new()
            .with_title("Hello, rusty world!")
            .with_dimensions(LogicalSize::new(cfg.width as f64, cfg.height as f64));
        let context = glutin::ContextBuilder::new()
            .with_vsync(cfg.vsync)
            .with_gl_debug_flag(true)
            .with_gl_profile(glutin::GlProfile::Compatibility) // nanovg doesn't work with Core.
            //.with_gl_profile(glutin::GlProfile::Core)
            .with_gl(glutin::GlRequest::Specific(glutin::Api::OpenGl, (4, 3)));
        let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();

        unsafe {
            gl_window.make_current().unwrap();
        }

        gl::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _);

        let mut vao: u32 = 0;

        unsafe {
            gl::DebugMessageCallback(Some(gl_debug_message), std::ptr::null_mut());

            // Disable everything by default
            gl::DebugMessageControl(
                gl::DONT_CARE,
                gl::DONT_CARE,
                gl::DONT_CARE,
                0,
                std::ptr::null_mut(),
                0,
            );

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

            gl::GenVertexArrays(1, &mut vao);
        }

        Rendertoy {
            events_loop,
            gl_window,
            mouse_state: MouseState::default(),
            cfg,
            keyboard: KeyboardState::new(),
            gl_state: GlState { vao },
            selected_debug_name: None,
            last_frame_instant: std::time::Instant::now(),
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
            .arg(
                clap::Arg::with_name("vsync")
                    .long("vsync")
                    .help("Wait for V-Sync")
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

        let mut keyboard_events: Vec<KeyboardInput> = Vec::new();
        let mut new_mouse_state = self.mouse_state.clone();

        for event in events.iter() {
            #[allow(clippy::single_match)]
            match event {
                glutin::Event::WindowEvent { event, .. } => match event {
                    glutin::WindowEvent::CloseRequested => running = false,
                    glutin::WindowEvent::Resized(logical_size) => {
                        let dpi_factor = self.gl_window.get_hidpi_factor();
                        let phys_size = logical_size.to_physical(dpi_factor);
                        self.gl_window.resize(phys_size);
                    }
                    glutin::WindowEvent::KeyboardInput { input, .. } => {
                        keyboard_events.push(*input);
                    }
                    glutin::WindowEvent::CursorMoved {
                        position: logical_pos,
                        device_id: _,
                        modifiers: _,
                    } => {
                        let dpi_factor = self.gl_window.get_hidpi_factor();
                        let pos = logical_pos.to_physical(dpi_factor);
                        new_mouse_state.pos = Point2::new(pos.x as f32, pos.y as f32);
                    }
                    glutin::WindowEvent::MouseInput { state, button, .. } => {
                        let button_id = match button {
                            glutin::MouseButton::Left => 0,
                            glutin::MouseButton::Middle => 1,
                            glutin::MouseButton::Right => 2,
                            _ => 0,
                        };

                        if let glutin::ElementState::Pressed = state {
                            new_mouse_state.button_mask |= 1 << button_id;
                        } else {
                            new_mouse_state.button_mask &= !(1 << button_id);
                        }
                    }
                    _ => (),
                },
                _ => (),
            }
        }

        // TODO: proper time
        self.keyboard.update(keyboard_events, 1.0 / 60.0);
        self.mouse_state.update(&new_mouse_state);

        running
    }

    fn draw_with_frame_snapshot<F>(&mut self, callback: &mut F)
    where
        F: FnMut(&FrameState) -> SnoozyRef<Texture>,
    {
        //unsafe {
        //gl::ClearColor(1.0, 1.0, 1.0, 1.0);
        //gl::Clear(gl::COLOR_BUFFER_BIT);
        //}

        let size = self
            .gl_window
            .get_inner_size()
            .map(|s| s.to_physical(self.gl_window.get_hidpi_factor()))
            .unwrap_or(glutin::dpi::PhysicalSize::new(1.0, 1.0));
        let window_size_pixels = (size.width as u32, size.height as u32);

        let now = std::time::Instant::now();
        let dt = now - self.last_frame_instant;
        self.last_frame_instant = now;

        let state = FrameState {
            mouse: &self.mouse_state,
            keys: &self.keyboard,
            window_size_pixels,
            dt: dt.as_secs_f32(),
        };

        let tex = callback(&state);

        with_snapshot(|snapshot| {
            let final_texture = &*snapshot.get(tex);

            let mut debugged_texture: Option<u32> = None;
            gpu_debugger::with_textures(|data| {
                debugged_texture = self
                    .selected_debug_name
                    .as_ref()
                    .and_then(|name| data.textures.get(name).cloned());
            });

            //draw_fullscreen_texture(final_texture, state.window_size_pixels);
            backend::draw::draw_fullscreen_texture(
                debugged_texture.unwrap_or(final_texture.texture_id),
                state.window_size_pixels,
            );
        });
    }

    fn draw_profiling_stats(&self, vg_context: &nanovg::Context, font: &Font) -> Option<String> {
        let size = self
            .gl_window
            .get_inner_size()
            .map(|s| s.to_physical(self.gl_window.get_hidpi_factor()))
            .unwrap_or(glutin::dpi::PhysicalSize::new(1.0, 1.0));

        let (width, height) = (size.width as i32, size.height as i32);

        unsafe {
            gl::Viewport(0, 0, width, height);
        }

        let (width, height) = (width as f32, height as f32);

        let mut selected_name = None;

        vg_context.frame(
            (width, height),
            self.gl_window.get_hidpi_factor() as f32,
            |frame| {
                let mut text_options = TextOptions {
                    size: 24.0,
                    color: Color::from_rgb(255, 255, 255),
                    align: Alignment::new().bottom().left(),
                    transform: None,
                    ..Default::default()
                };

                let mut text_shadow_options = text_options.clone();
                text_shadow_options.color = Color::from_rgb(0, 0, 0);
                text_shadow_options.blur = 1.0;

                let metrics = frame.text_metrics(*font, text_options);

                let mut y = 10.0 + metrics.line_height;

                gpu_profiler::with_stats(|stats| {
                    /*for (name, scope) in stats.scopes.iter() {
                        let text = format!("{}: {:.3}ms", name, scope.average_duration_millis());
                        let (uw, _) = frame.text_bounds(*font, (0.0, 0.0), &text, text_options);

                        // self.mouse_state.pos.y
                        let hit = self.mouse_state.pos.y >= (y - metrics.line_height)
                            && self.mouse_state.pos.y < y
                            && self.mouse_state.pos.x < uw + 10.0;
                        if hit {
                            selected_name = Some(name.to_owned());
                        }

                        let color = if hit {
                            Color::from_rgb(255, 64, 16)
                        } else {
                            Color::from_rgb(255, 255, 255)
                        };
                        text_options.color = color;

                        frame.text(*font, (10.0 + 1.0, y + 1.0), &text, text_shadow_options);
                        frame.text(*font, (10.0, y), &text, text_options);
                        y += metrics.line_height;
                    }*/

                    for name in stats.order.iter() {
                        if let Some(scope) = stats.scopes.get(name) {
                            let text =
                                format!("{}: {:.3}ms", name, scope.average_duration_millis());
                            let (uw, _) = frame.text_bounds(*font, (0.0, 0.0), &text, text_options);

                            // self.mouse_state.pos.y
                            let hit = self.mouse_state.pos.y >= (y - metrics.line_height)
                                && self.mouse_state.pos.y < y
                                && self.mouse_state.pos.x < uw + 10.0;
                            if hit {
                                selected_name = Some(name.to_owned());
                            }

                            let color = if hit {
                                Color::from_rgb(255, 64, 16)
                            } else {
                                Color::from_rgb(255, 255, 255)
                            };
                            text_options.color = color;

                            frame.text(*font, (10.0 + 1.0, y + 1.0), &text, text_shadow_options);
                            frame.text(*font, (10.0, y), &text, text_options);
                            y += metrics.line_height;
                        }
                    }
                });
            },
        );

        selected_name
    }

    pub fn draw_forever<F>(&mut self, mut callback: F)
    where
        F: FnMut(&FrameState) -> SnoozyRef<Texture>,
    {
        let vg_context = nanovg::ContextBuilder::new()
            .stencil_strokes()
            .build()
            .expect("Initialization of NanoVG failed!");

        let mut font = None;
        with_snapshot(|snapshot| {
            let blob = &*snapshot.get(load_blob(asset!("fonts/Roboto-Regular.ttf")));

            font = Some(
                Font::from_memory(&vg_context, "Roboto-Regular", blob.contents.as_slice())
                    .expect("Failed to load font 'Roboto-Regular.ttf'"),
            );
        });

        let font = font.expect("Failed to load font");

        let mut running = true;
        while running {
            unsafe {
                gl::ClipControl(gl::LOWER_LEFT, gl::ZERO_TO_ONE);
                gl::BindVertexArray(self.gl_state.vao);
                gl::Enable(gl::CULL_FACE);
                gl::Disable(gl::STENCIL_TEST);
                gl::Disable(gl::BLEND);
            }

            self.draw_with_frame_snapshot(&mut callback);
            gpu_profiler::end_frame();
            gpu_debugger::end_frame();

            self.selected_debug_name = self.draw_profiling_stats(&vg_context, &font);

            running = self.next_frame();
        }
    }
}

pub fn draw_fullscreen_texture(tex: &Texture, framebuffer_size: (u32, u32)) {
    backend::draw::draw_fullscreen_texture(tex.texture_id, framebuffer_size);
}

impl TextureKey {
    pub fn fullscreen(rtoy: &Rendertoy, format: u32) -> Self {
        TextureKey {
            width: rtoy.width(),
            height: rtoy.height(),
            format,
        }
    }
}
