#![feature(core_intrinsics)]
#![feature(async_closure)]

mod backend;
mod blob;
mod buffer;
mod camera;
mod consts;
mod gpu_debugger;
mod gpu_profiler;
mod gui;
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

#[global_allocator]
static ALLOC: rpmalloc::RpMalloc = rpmalloc::RpMalloc;

use clap::ArgMatches;
use gpu_profiler::GpuProfilerStats;
use gui::ImGuiBackend;
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
    dt: f32,
    show_gui: bool,
    average_frame_time: f32,
    frame_time_display_cooldown: f32,
    imgui: imgui::Context,
    imgui_backend: ImGuiBackend,
    gpu_profiler_stats: Option<GpuProfilerStats>,
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

        RendertoyConfig {
            width,
            height,
            vsync,
            graphics_debugging,
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

        crate::vulkan::initialize_vk_state(
            VkKitchenSink::new(&window, cfg.graphics_debugging).unwrap(),
        );

        let mut imgui = imgui::Context::create();
        let imgui_backend = ImGuiBackend::new(&window, &mut imgui);

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
            dt: 0.0,
            show_gui: true,
            average_frame_time: 0.0,
            frame_time_display_cooldown: 0.0,
            imgui,
            imgui_backend,
            gpu_profiler_stats: None,
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
        {
            let imgui_backend = &mut self.imgui_backend;
            let imgui = &mut self.imgui;
            let window = &self.window;

            self.events_loop.poll_events(|event| {
                imgui_backend.handle_event(window, imgui, &event);
                events.push(event);
            });
        }

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
                        /*winit::WindowEvent::Resized(logical_size) => {
                            let dpi_factor = self.window.get_hidpi_factor();
                            let phys_size = logical_size.to_physical(dpi_factor);
                            windowed_context.resize(phys_size);
                        }*/
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

    fn get_currently_debugged_texture(&self) -> Option<String> {
        self.selected_debug_name
            .clone()
            .or(self.locked_debug_name.clone())
    }

    fn draw_with_frame_snapshot<F>(&mut self, callback: &mut F) -> vk::ImageView
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
        self.dt = dt.as_secs_f32();

        self.average_frame_time = if 0.0f32 == self.average_frame_time {
            self.dt
        } else {
            let blend = (-4.0 * self.dt).exp();
            self.average_frame_time * blend + self.dt * (1.0 - blend)
        };

        self.frame_time_display_cooldown += self.dt;
        if self.frame_time_display_cooldown > 1.0 {
            self.frame_time_display_cooldown = 0.0;
            println!(
                "frame time: {:.2}ms ({:.2} fps)",
                self.average_frame_time * 1000.0,
                1.0 / self.average_frame_time
            );
        }

        let state = FrameState {
            mouse: &self.mouse_state,
            keys: &self.keyboard,
            window_size_pixels,
            dt: self.dt,
        };

        let tex = callback(&state);

        let final_texture = self.rt.block_on(async move {
            let snapshot = get_snapshot();
            let final_texture: Texture = (*snapshot.get(tex).await).clone();
            final_texture.view
        });

        //with_gl(|gl| unsafe
        let debugged_texture = {
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
                    .and_then(|name| data.textures.get(&name).cloned());
            });
            debugged_texture

            // TODO
            //draw_fullscreen_texture(final_texture, state.window_size_pixels);

            // TODO: copy output to screen
            /*backend::draw::draw_fullscreen_texture(
                &gfx,
                debugged_texture.unwrap_or(final_texture.view),
                state.window_size_pixels,
            );*/
        };
        //);
        //
        debugged_texture.unwrap_or(final_texture)
    }

    fn draw_profiling_stats(
        ui: &imgui::Ui,
        average_frame_time: f32,
        stats: &GpuProfilerStats,
        currently_debugged_texture: &Option<String>,
    ) -> Option<String> {
        let mut selected_name = None;
        let fps = format!("FPS: {:.1}", 1.0 / average_frame_time);
        ui.text(fps);
        //let mut total_time_ms = 0.0;

        for (name, scope) in stats.scopes.iter() {
            let text = format!("{}: {:.3}ms", name, scope.average_duration_millis());

            let style = if Some(name) == currently_debugged_texture.as_ref() {
                Some(ui.push_style_color(imgui::StyleColor::Text, [1.0, 0.25, 0.0625, 1.0]))
            } else {
                None
            };

            ui.text(text);

            if let Some(style) = style {
                style.pop(ui);
            }

            let hit = ui.is_item_hovered();
            if hit {
                selected_name = Some(name.to_owned());
            }
        }

        /*for name in stats.order.iter() {
            if let Some(scope) = stats.scopes.get(name) {
                let average_duration_millis = scope.average_duration_millis();
                let text = format!("{}: {:.3}ms", name, average_duration_millis);
                total_time_ms += average_duration_millis;

                /*let color = if Some(name) == self.get_currently_debugged_texture() {
                    Color::from_rgb(255, 64, 16)
                } else {
                    Color::from_rgb(255, 255, 255)
                };
                text_options.color = color;*/

                ui.text(text);

                if ui.is_item_hovered() {
                    selected_name = Some(name.to_owned());
                }

                /*if (self.mouse_state.button_mask & 1) != 0 {
                    self.locked_debug_name = selected_name.clone();
                }*/
            }
        }*/

        //let text = format!("TOTAL: {:.3}ms", total_time_ms);
        //ui.text(text);

        selected_name
    }

    /*fn draw_warnings(
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

    pub fn draw_forever<F>(mut self, mut callback: F)
    where
        F: FnMut(&FrameState) -> SnoozyRef<Texture>,
    {
        use ash::version::DeviceV1_0;
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
                        .bindings(&[
                            vk::DescriptorSetLayoutBinding::builder()
                                .descriptor_count(1)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                                .binding(0)
                                .build(),
                            vk::DescriptorSetLayoutBinding::builder()
                                .descriptor_count(1)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                                .binding(1)
                                .build(),
                            vk::DescriptorSetLayoutBinding::builder()
                                .descriptor_count(1)
                                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                                .binding(2)
                                .build(),
                            vk::DescriptorSetLayoutBinding::builder()
                                .descriptor_count(1)
                                .descriptor_type(vk::DescriptorType::SAMPLER)
                                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                                .binding(3)
                                .immutable_samplers(&[vk.samplers[vulkan::SAMPLER_LINEAR]])
                                .build(),
                        ])
                        .build(),
                    None,
                )
                .unwrap()
        };

        let present_descriptor_sets =
            vk.create_present_descriptor_sets(present_descriptor_set_layout);
        let present_pipeline =
            create_present_compute_pipeline(&vk.device, present_descriptor_set_layout)
                .expect("create_present_compute_pipeline");

        let mut running = true;
        while running {
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
                |_device| {
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

                    let currently_debugged_texture = self.get_currently_debugged_texture().clone();
                    let ui =
                        self.imgui_backend
                            .prepare_frame(&self.window, &mut self.imgui, self.dt);
                    {
                        //ui.separator();

                        if self.show_gui {
                            /*self.draw_warnings(
                                gl,
                                windowed_context,
                                &vg_context,
                                &font,
                                RTOY_WARNINGS.lock().unwrap().drain(..),
                            );*/
                            if let Some(ref stats) = self.gpu_profiler_stats {
                                self.selected_debug_name = Self::draw_profiling_stats(
                                    &ui,
                                    self.average_frame_time,
                                    stats,
                                    &currently_debugged_texture,
                                );
                            }
                        }
                    }
                    let gui_texture_view = self.imgui_backend.render(&self.window, ui, cb);

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
                                        .image_view(final_texture)
                                        .build()])
                                    .build(),
                                vk::WriteDescriptorSet::builder()
                                    .dst_set(present_descriptor_sets[present_index])
                                    .dst_binding(1)
                                    .dst_array_element(0)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .image_info(&[vk::DescriptorImageInfo::builder()
                                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                        .image_view(gui_texture_view)
                                        .build()])
                                    .build(),
                                vk::WriteDescriptorSet::builder()
                                    .dst_set(present_descriptor_sets[present_index])
                                    .dst_binding(2)
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
                        let push_constants: (f32, f32) =
                            (1.0 / vk.window_width as f32, 1.0 / vk.window_height as f32);
                        vk.device.cmd_push_constants(
                            cb,
                            present_pipeline.pipeline_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            std::slice::from_raw_parts(
                                &push_constants.0 as *const f32 as *const u8,
                                2 * 4,
                            ),
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

            //with_gl_and_context(|gl, windowed_context|
            {
                let gfx = Gfx {}; // TODO
                gpu_profiler::end_frame(&gfx);
                gpu_debugger::end_frame();

                self.gpu_profiler_stats = Some(gpu_profiler::get_stats());
            }
            //);

            running = self.next_frame();
        }
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
    use ash::version::DeviceV1_0;
    use std::ffi::CString;
    use std::io::Cursor;

    let shader_entry_name = CString::new("main").unwrap();
    let mut shader_spv = Cursor::new(&include_bytes!("final_blit.spv")[..]);
    let shader_code = ash::util::read_spv(&mut shader_spv).expect("Failed to read shader spv");

    let descriptor_set_layouts = [descriptor_set_layout];
    let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&descriptor_set_layouts)
        .push_constant_ranges(&[vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 2 * 4,
        }]);

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
