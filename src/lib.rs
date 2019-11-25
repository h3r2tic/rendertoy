#![feature(core_intrinsics)]
#![feature(async_closure)]

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
mod switchable_graphics;
mod texture;
mod viewport;
mod vulkan;

pub mod compute_tex_macro;

pub use nalgebra as na;
pub use snoozy::*;
use tokio::runtime::Runtime;

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
pub use self::vulkan::VkKitchenSink;

pub use ash::{vk, vk::Format};

pub type Point2 = na::Point2<f32>;
pub type Vector2 = na::Vector2<f32>;

pub type Point3 = na::Point3<f32>;
pub type Vector3 = na::Vector3<f32>;

pub type Point4 = na::Point4<f32>;
pub type Vector4 = na::Vector4<f32>;

pub type Matrix3 = na::Matrix3<f32>;
pub type Matrix4 = na::Matrix4<f32>;
pub type Isometry3 = na::Isometry3<f32>;

pub type Quaternion = na::Quaternion<f32>;
pub type UnitQuaternion = na::UnitQuaternion<f32>;

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
use std::str::FromStr;

/*extern "system" fn gl_debug_message(
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

        let is_ignored_id = match id {
            131216 => true, // Program/shader state info: GLSL shader * failed to compile. WAT.
            131185 => true, // Buffer detailed info: (...) will use (...) memory as the source for buffer object operations.
            _ => false,
        };

        if !is_ignored_id {
            let is_important_type = match type_ {
                gl::DEBUG_TYPE_ERROR
                | gl::DEBUG_TYPE_UNDEFINED_BEHAVIOR
                | gl::DEBUG_TYPE_DEPRECATED_BEHAVIOR
                | gl::DEBUG_TYPE_PORTABILITY => true,
                _ => false,
            };

            if !is_important_type {
                println!("GL debug({}): {}\n", id, s.to_string_lossy());
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
}

struct GlState {
    vao: u32,
}*/

pub struct Rendertoy {
    rt: Runtime,
    window: winit::Window,
    events_loop: winit::EventsLoop,
    mouse_state: MouseState,
    cfg: RendertoyConfig,
    keyboard: KeyboardState,
    selected_debug_name: Option<String>,
    locked_debug_name: Option<String>,
    last_frame_instant: std::time::Instant,
    show_gui: bool,
    average_frame_time: f32,
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
    pub graphics_debugging: bool,
    pub core_gl: bool,
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

        let graphics_debugging = !matches.is_present("ndebug");
        let core_gl = matches.is_present("core-gl");

        RendertoyConfig {
            width,
            height,
            vsync,
            graphics_debugging,
            core_gl,
        }
    }
}

impl Rendertoy {
    pub fn new_with_config(cfg: RendertoyConfig) -> Rendertoy {
        let rt = Runtime::new().unwrap();
        let events_loop = winit::EventsLoop::new();

        let window = winit::WindowBuilder::new()
            .with_title("Rendertoy")
            .with_dimensions(winit::dpi::LogicalSize::new(
                cfg.width as f64,
                cfg.height as f64,
            ))
            .build(&events_loop)
            .expect("window");

        crate::vulkan::initialize_vk_state(VkKitchenSink::new(&window).unwrap());

        /*let windowed_context = winit::ContextBuilder::new()
            .with_vsync(cfg.vsync)
            .with_gl_debug_flag(cfg.graphics_debugging)
            .with_gl_profile(if cfg.core_gl {
                winit::GlProfile::Core
            } else {
                winit::GlProfile::Compatibility
            })
            .with_gl(winit::GlRequest::Specific(winit::Api::OpenGl, (4, 3)))
            .build_windowed(wb, &events_loop)
            .unwrap();

        let windowed_context = unsafe { windowed_context.make_current().unwrap() };*/
        //let gl = gl::Gl::load_with(|symbol| windowed_context.get_proc_address(symbol) as *const _);
        //let mut vao: u32 = 0;

        /*unsafe {
            use std::ffi::CStr;
            println!(
                "GL_VENDOR: {:?}",
                CStr::from_ptr(gl.GetString(gl::VENDOR) as *const i8)
            );
            println!(
                "GL_RENDERER: {:?}",
                CStr::from_ptr(gl.GetString(gl::RENDERER) as *const i8)
            );

            gl.DebugMessageCallback(Some(gl_debug_message), std::ptr::null_mut());

            // Disable everything by default
            gl.DebugMessageControl(
                gl::DONT_CARE,
                gl::DONT_CARE,
                gl::DONT_CARE,
                0,
                std::ptr::null_mut(),
                0,
            );

            gl.DebugMessageControl(
                gl::DONT_CARE,
                gl::DONT_CARE,
                gl::DONT_CARE,
                0,
                std::ptr::null_mut(),
                1,
            );

            gl.DebugMessageControl(
                gl::DEBUG_SOURCE_SHADER_COMPILER,
                gl::DONT_CARE,
                gl::DONT_CARE,
                0,
                std::ptr::null_mut(),
                0,
            );

            gl.Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl.Enable(gl::FRAMEBUFFER_SRGB);

            gl.GenVertexArrays(1, &mut vao);
        }*/

        //set_global_gl_context(gfx, unsafe { windowed_context.make_not_current().unwrap() });
        //set_global_gl_context(gfx, unsafe { windowed_context.treat_as_not_current() });

        Rendertoy {
            rt,
            window,
            events_loop,
            mouse_state: MouseState::default(),
            cfg,
            keyboard: KeyboardState::new(),
            selected_debug_name: None,
            locked_debug_name: None,
            last_frame_instant: std::time::Instant::now(),
            show_gui: true,
            average_frame_time: 0.0,
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
            .arg(
                clap::Arg::with_name("ndebug")
                    .long("ndebug")
                    .help("Disable graphics debugging"),
            )
            .arg(
                clap::Arg::with_name("core-gl")
                    .long("core-gl")
                    .help("Use the Core profile of OpenGL (disables UI)"),
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
        let mut events = Vec::new();
        self.events_loop.poll_events(|event| events.push(event));

        //with_gl_and_context(|_, windowed_context|
        {
            //windowed_context.swap_buffers().unwrap();

            let mut running = true;

            let mut keyboard_events: Vec<KeyboardInput> = Vec::new();
            let mut new_mouse_state = self.mouse_state.clone();

            for event in events.iter() {
                #[allow(clippy::single_match)]
                match event {
                    winit::Event::WindowEvent { event, .. } => match event {
                        winit::WindowEvent::CloseRequested => running = false,
                        winit::WindowEvent::Resized(logical_size) => {
                            let dpi_factor = self.window.get_hidpi_factor();
                            let phys_size = logical_size.to_physical(dpi_factor);

                            //windowed_context.resize(phys_size);
                        }
                        winit::WindowEvent::KeyboardInput { input, .. } => {
                            if input.virtual_keycode == Some(VirtualKeyCode::Tab) {
                                if input.state == ElementState::Pressed {
                                    self.show_gui = !self.show_gui;
                                }
                            } else {
                                keyboard_events.push(*input);
                            }
                        }
                        winit::WindowEvent::CursorMoved {
                            position: logical_pos,
                            device_id: _,
                            modifiers: _,
                        } => {
                            let dpi_factor = self.window.get_hidpi_factor();
                            let pos = logical_pos.to_physical(dpi_factor);
                            new_mouse_state.pos = Point2::new(pos.x as f32, pos.y as f32);
                        }
                        winit::WindowEvent::MouseInput { state, button, .. } => {
                            let button_id = match button {
                                winit::MouseButton::Left => 0,
                                winit::MouseButton::Middle => 1,
                                winit::MouseButton::Right => 2,
                                _ => 0,
                            };

                            if let winit::ElementState::Pressed = state {
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
        //)
    }

    fn get_currently_debugged_texture(&self) -> Option<&String> {
        self.selected_debug_name
            .as_ref()
            .or(self.locked_debug_name.as_ref())
    }

    fn draw_with_frame_snapshot<F>(&mut self, callback: &mut F) -> Texture
    where
        F: FnMut(&FrameState) -> SnoozyRef<Texture>,
    {
        let size = self
            .window
            .get_inner_size()
            .map(|s| s.to_physical(self.window.get_hidpi_factor()))
            .unwrap_or(winit::dpi::PhysicalSize::new(1.0, 1.0));
        let window_size_pixels = (size.width as u32, size.height as u32);

        let now = std::time::Instant::now();
        let dt = now - self.last_frame_instant;
        self.last_frame_instant = now;

        self.average_frame_time = if 0.0f32 == self.average_frame_time {
            dt.as_secs_f32()
        } else {
            let dt = dt.as_secs_f32();
            let blend = (-4.0 * dt).exp();
            self.average_frame_time * blend + dt * (1.0 - blend)
        };

        let state = FrameState {
            mouse: &self.mouse_state,
            keys: &self.keyboard,
            window_size_pixels,
            dt: dt.as_secs_f32(),
        };

        let tex = callback(&state);

        let final_texture = self.rt.block_on(async move {
            let snapshot = get_snapshot();
            let final_texture: Texture = (*snapshot.get(tex).await).clone();
            final_texture
        });

        //with_gl(|gl| unsafe
        {
            /*gl.ClipControl(gl::LOWER_LEFT, gl::ZERO_TO_ONE);
            gl.BindVertexArray(self.gl_state.vao);
            gl.Enable(gl::CULL_FACE);
            gl.Disable(gl::STENCIL_TEST);
            gl.Disable(gl::BLEND);*/

            let mut debugged_texture: Option<vk::ImageView> = None;
            gpu_debugger::with_textures(|data| {
                /*debugged_texture = self
                .selected_debug_name
                .as_ref()
                .or(self.locked_debug_name.as_ref())
                .and_then(|name| data.textures.get(name).cloned());*/
                debugged_texture = self
                    .get_currently_debugged_texture()
                    .and_then(|name| data.textures.get(name).cloned());
            });

            // TODO
            let gfx = Gfx {};
            //draw_fullscreen_texture(final_texture, state.window_size_pixels);

            // TODO: copy output to screen
            /*backend::draw::draw_fullscreen_texture(
                &gfx,
                debugged_texture.unwrap_or(final_texture.view),
                state.window_size_pixels,
            );*/
        }
        //);
        // 
        final_texture
    }

    /*fn draw_profiling_stats(
        &mut self,
        gfx: &crate::Gfx,
        windowed_context: &GlutinCurrentContext,
        vg_context: &nanovg::Context,
        font: &Font,
    ) -> Option<String> {
        let size = windowed_context
            .window()
            .get_inner_size()
            .map(|s| s.to_physical(windowed_context.window().get_hidpi_factor()))
            .unwrap_or(winit::dpi::PhysicalSize::new(1.0, 1.0));

        let (width, height) = (size.width as i32, size.height as i32);

        unsafe {
            gl.Viewport(0, 0, width, height);
        }

        let (width, height) = (width as f32, height as f32);

        let mut selected_name = None;

        vg_context.frame(
            (width, height),
            windowed_context.window().get_hidpi_factor() as f32,
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

                let fps = format!("FPS: {:.1}", 1.0 / self.average_frame_time);

                let metrics = frame.text_metrics(*font, text_options);
                let mut y = 10.0 + metrics.line_height;
                frame.text(*font, (10.0 + 1.0, y + 1.0), &fps, text_shadow_options);
                frame.text(*font, (10.0, y), &fps, text_options);
                y += metrics.line_height * 2.0;

                let mut total_time_ms = 0.0;

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
                            let average_duration_millis = scope.average_duration_millis();
                            let text = format!("{}: {:.3}ms", name, average_duration_millis);
                            total_time_ms += average_duration_millis;

                            let (uw, _) = frame.text_bounds(*font, (0.0, 0.0), &text, text_options);

                            // self.mouse_state.pos.y
                            let hit = self.mouse_state.pos.y >= (y - metrics.line_height)
                                && self.mouse_state.pos.y < y
                                && self.mouse_state.pos.x < uw + 10.0;
                            if hit {
                                selected_name = Some(name.to_owned());
                            }

                            if (self.mouse_state.button_mask & 1) != 0 {
                                self.locked_debug_name = selected_name.clone();
                            }

                            let color = if Some(name) == self.get_currently_debugged_texture() {
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

                let text = format!("TOTAL: {:.3}ms", total_time_ms);

                let color = Color::from_rgb(255, 255, 255);
                text_options.color = color;

                frame.text(*font, (10.0 + 1.0, y + 1.0), &text, text_shadow_options);
                frame.text(*font, (10.0, y), &text, text_options);
            },
        );

        selected_name
    }

    fn draw_warnings(
        &self,
        gfx: &crate::Gfx,
        windowed_context: &GlutinCurrentContext,
        vg_context: &nanovg::Context,
        font: &Font,
        warnings: impl Iterator<Item = String>,
    ) {
        let size = windowed_context
            .window()
            .get_inner_size()
            .map(|s| s.to_physical(windowed_context.window().get_hidpi_factor()))
            .unwrap_or(winit::dpi::PhysicalSize::new(1.0, 1.0));

        let (width, height) = (size.width as i32, size.height as i32);

        unsafe {
            gl.Viewport(0, 0, width, height);
        }

        let (width, height) = (width as f32, height as f32);

        vg_context.frame(
            (width, height),
            windowed_context.window().get_hidpi_factor() as f32,
            |frame| {
                let text_options = TextOptions {
                    size: 24.0,
                    color: Color::from_rgb(255, 16, 4),
                    align: Alignment::new().bottom().left(),
                    transform: None,
                    ..Default::default()
                };

                let mut text_shadow_options = text_options.clone();
                text_shadow_options.color = Color::from_rgb(0, 0, 0);
                text_shadow_options.blur = 1.0;

                let metrics = frame.text_metrics(*font, text_options);

                let warnings = warnings.take(8).collect::<Vec<_>>();

                let mut y = height - metrics.line_height * (warnings.len() as f32);
                for line in warnings {
                    frame.text(*font, (10.0 + 1.0, y + 1.0), &line, text_shadow_options);
                    frame.text(*font, (10.0, y), &line, text_options);
                    y += metrics.line_height;
                }
            },
        );
    }*/

    pub fn draw_forever<F>(mut self, mut callback: F) -> snoozy::Result<()>
    where
        F: FnMut(&FrameState) -> SnoozyRef<Texture>,
    {
        use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0, InstanceV1_1};
        use vulkan::*;

        /*let vg_context = with_gl(|_| {
            nanovg::ContextBuilder::new()
                .stencil_strokes()
                .build()
                .expect("Initialization of NanoVG failed!")
        });

        let font = with_gl(|_| {
            let snapshot = get_snapshot();
            let rt = Runtime::new().unwrap();
            let blob = &*rt.block_on(snapshot.get(load_blob(asset!("fonts/Roboto-Regular.ttf"))));

                Font::from_memory(&vg_context, "Roboto-Regular", blob.contents.as_slice())
                    .expect("Failed to load font 'Roboto-Regular.ttf'")
        });

        dbg!();*/

        let mut swapchain_acquired_semaphore_idx = 0;

        let vk = unsafe { vk_all() };
        let present_descriptor_set_layout = unsafe {
            vk.device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder()
                        .bindings(&[vk::DescriptorSetLayoutBinding::builder()
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .binding(0)
                            .build(), 
                            vk::DescriptorSetLayoutBinding::builder()
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .binding(1)
                            .build()])
                        .build(),
                    None,
                )
                .unwrap()
        };

        let present_descriptor_sets =
            vk.create_present_descriptor_sets(present_descriptor_set_layout);
        let present_pipeline =
            create_present_compute_pipeline(&vk.device, present_descriptor_set_layout)?;

        let mut running = true;
        while running {
            // TODO: reset descriptor pool for the current frame
            // TODO: reset atomics in dynamic uniform buffers
            // TODO: vk_begin_frame(present_image_index);

            swapchain_acquired_semaphore_idx =
                (swapchain_acquired_semaphore_idx + 1) % vk.frame_data.len();

            let present_index = unsafe {
                let (present_index, _) = vk
                    .swapchain_loader
                    .acquire_next_image(
                        vk.swapchain,
                        std::u64::MAX,
                        vk.swapchain_acquired_semaphores[swapchain_acquired_semaphore_idx],
                        vk::Fence::null(),
                    )
                    .unwrap();
                present_index as usize
            };
            unsafe {
                vk_begin_frame(present_index);
            }

            let vk_frame = unsafe { vk_frame() };

            // TODO: don't abuse the Copy
            let cb = vk_frame.command_buffer.lock().unwrap().cb;

            record_submit_commandbuffer(
                &vk.device,
                cb,
                vk.present_queue,
                &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                &[vk.swapchain_acquired_semaphores[swapchain_acquired_semaphore_idx]],
                &[vk_frame.rendering_complete_semaphore],
                |_device, command_buffer| {
                    let present_image = vk_frame.present_image;

                    vk.record_image_barrier(
                        cb,
                        ImageBarrier::new(
                            present_image,
                            vk_sync::AccessType::Present,
                            vk_sync::AccessType::ComputeShaderWrite,
                        )
                        .with_discard(true),
                    );

                    let final_texture = self.draw_with_frame_snapshot(&mut callback);

                    unsafe {
                        vk.device.update_descriptor_sets(
                            &[
                                vk::WriteDescriptorSet::builder()
                                    .dst_set(present_descriptor_sets[present_index])
                                    .dst_binding(0)
                                    .dst_array_element(0)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .image_info(&[vk::DescriptorImageInfo::builder()
                                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                        .image_view(final_texture.view)
                                        .build()])
                                    .build(),
                                vk::WriteDescriptorSet::builder()
                                    .dst_set(present_descriptor_sets[present_index])
                                    .dst_binding(1)
                                    .dst_array_element(0)
                                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                    .image_info(&[vk::DescriptorImageInfo::builder()
                                        .image_layout(vk::ImageLayout::GENERAL)
                                        .image_view(vk_frame.present_image_view)
                                        .build()])
                                    .build(),
                            ],
                            &[],
                        );
        
                        vk.device.cmd_bind_pipeline(
                            cb,
                            vk::PipelineBindPoint::COMPUTE,
                            present_pipeline.pipeline,
                        );
                        vk.device.cmd_bind_descriptor_sets(
                            cb,
                            vk::PipelineBindPoint::COMPUTE,
                            present_pipeline.pipeline_layout,
                            0,
                            &[present_descriptor_sets[present_index]],
                            &[],
                        );
                        vk.device
                            .cmd_dispatch(cb, vk.window_width / 8, vk.window_height / 8, 1);
                    }

                    vk.record_image_barrier(
                        cb,
                        ImageBarrier::new(
                            present_image,
                            vk_sync::AccessType::ComputeShaderWrite,
                            vk_sync::AccessType::Present,
                        ),
                    );
                },
            );

            let wait_semaphores = [vk_frame.rendering_complete_semaphore];
            let swapchains = [vk.swapchain];
            let image_indices = [present_index as u32];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            unsafe {
                vk.swapchain_loader
                    .queue_present(vk.present_queue, &present_info)
                    .unwrap();
            }

            // TODO: flush mapped uniform buffer ranges
            // TODO: submit main command buffer

            //with_gl_and_context(|gl, windowed_context|
            {
                let gfx = Gfx {}; // TODO
                gpu_profiler::end_frame(&gfx);
                gpu_debugger::end_frame();

                /*if self.show_gui && !self.cfg.core_gl {
                    self.draw_warnings(
                        gl,
                        windowed_context,
                        &vg_context,
                        &font,
                        RTOY_WARNINGS.lock().unwrap().drain(..),
                    );
                    self.selected_debug_name =
                        self.draw_profiling_stats(gfx, windowed_context, &vg_context, &font);
                }*/
            }
            //);

            running = self.next_frame();
        }

        Ok(())
    }
}

/*pub fn draw_fullscreen_texture(tex: &Texture, framebuffer_size: (u32, u32)) {
    backend::draw::draw_fullscreen_texture(tex.texture_id, framebuffer_size);
}*/

lazy_static! {
    static ref RTOY_WARNINGS: std::sync::Mutex<Vec<String>> =
        { std::sync::Mutex::new(Default::default()) };
}

pub fn rtoy_show_warning(text: String) {
    RTOY_WARNINGS.lock().unwrap().push(text);
}

impl TextureKey {
    pub fn fullscreen(rtoy: &Rendertoy, format: vk::Format) -> Self {
        TextureKey {
            width: rtoy.width(),
            height: rtoy.height(),
            format: format.as_raw(),
        }
    }
}

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

// TODO
pub struct Gfx {}

fn create_present_compute_pipeline(
    vk_device: &ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> snoozy::Result<crate::shader::ComputePipeline> {
    use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0, InstanceV1_1};
    use std::ffi::{CStr, CString};
    use std::io::Cursor;

    let shader_entry_name = CString::new("main").unwrap();
    let mut shader_spv = Cursor::new(&include_bytes!("copy_image.spv")[..]);
    let shader_code = ash::util::read_spv(&mut shader_spv).expect("Failed to read shader spv");

    let descriptor_set_layouts = [descriptor_set_layout];
    let layout_create_info =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);

    unsafe {
        let shader_module = vk_device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::builder().code(&shader_code),
                None,
            )
            .unwrap();

        let stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(shader_module)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(&shader_entry_name);

        let pipeline_layout = vk_device
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_create_info.build())
            .layout(pipeline_layout);

        // TODO: pipeline cache
        let pipeline = vk_device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None)
            .expect("pipeline")[0];

        /*let data = ComputePipelineData {
            pipeline: pipelines[0],
            descriptor_layouts,
            layout: pipeline_layout,
        };*/

        Ok(crate::shader::ComputePipeline {
            pipeline_layout,
            pipeline,
        })
    }
}
