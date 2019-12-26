use crate::gpu_debugger;
use crate::gpu_profiler::GpuProfilerStats;
use crate::gui::ImGuiBackend;
use crate::keyboard::*;
use crate::renderer::{RenderFrameStatus, Renderer};
use crate::texture::{Texture, TextureKey};
use crate::vulkan;
use crate::{Point2, Vector2};
use ash::vk;
use clap::ArgMatches;
use imgui::im_str;
use snoozy::{get_snapshot, OpaqueSnoozyRef, Result, SnoozyRef};
use std::str::FromStr;
use std::sync::Arc;
use tokio::runtime::Runtime;

struct RendertoyState {
    rt: Runtime,
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
    dump_next_frame_dot_graph: bool,
    initialization_instant: std::time::Instant,
    time_to_first_frame: Option<std::time::Duration>,
}

pub struct Rendertoy {
    state: RendertoyState,
    renderer: crate::renderer::Renderer,
    imgui_backend: ImGuiBackend,
    gui_placeholder_tex: Texture,
    imgui: imgui::Context,
    window: Arc<winit::Window>,
    events_loop: winit::EventsLoop,
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
    pub fn new_with_config(cfg: RendertoyConfig) -> Self {
        let rt = Runtime::new().unwrap();
        tracing_subscriber::fmt::init();

        let events_loop = winit::EventsLoop::new();

        let window = winit::WindowBuilder::new()
            .with_title("Rendertoy")
            .with_dimensions(winit::dpi::LogicalSize::new(
                cfg.width as f64,
                cfg.height as f64,
            ))
            .build(&events_loop)
            .expect("window");
        let window = Arc::new(window);

        let renderer = Renderer::new(window.clone(), cfg.graphics_debugging, cfg.vsync);

        let mut imgui = imgui::Context::create();
        let mut imgui_backend = ImGuiBackend::new(&window, &mut imgui);
        imgui_backend.create_graphics_resources();

        let gui_placeholder_tex = {
            let texel_value = [0u8; 4];
            let image_dimensions = (1, 1);
            let internal_format = vk::Format::R8G8B8A8_UNORM;

            crate::texture::load_tex_impl(&texel_value, image_dimensions, internal_format)
                .expect("gui placeholder texture")
        };

        Rendertoy {
            state: RendertoyState {
                rt,
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
                dump_next_frame_dot_graph: false,
                initialization_instant: std::time::Instant::now(),
                time_to_first_frame: None,
            },
            renderer,
            imgui_backend,
            gui_placeholder_tex,
            imgui,
            window,
            events_loop,
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
        self.state.cfg.width
    }

    pub fn height(&self) -> u32 {
        self.state.cfg.height
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

        let mut running = true;

        let mut keyboard_events: Vec<KeyboardInput> = Vec::new();
        let mut new_mouse_state = self.state.mouse_state.clone();

        let gui_want_capture_mouse = self.imgui.io().want_capture_mouse;

        for event in events.iter() {
            #[allow(clippy::single_match)]
            match event {
                winit::Event::WindowEvent { event, .. } => match event {
                    winit::WindowEvent::CloseRequested => running = false,
                    winit::WindowEvent::KeyboardInput { input, .. } => {
                        if input.virtual_keycode == Some(VirtualKeyCode::Tab) {
                            if input.state == ElementState::Pressed {
                                self.state.show_gui = !self.state.show_gui;
                            }
                        } else {
                            keyboard_events.push(*input);
                        }
                    }
                    winit::WindowEvent::CursorMoved {
                        position: logical_pos,
                        device_id: _,
                        modifiers: _,
                    } if !gui_want_capture_mouse => {
                        let dpi_factor = self.window.get_hidpi_factor();
                        let pos = logical_pos.to_physical(dpi_factor);
                        new_mouse_state.pos = Point2::new(pos.x as f32, pos.y as f32);
                    }
                    winit::WindowEvent::MouseInput { state, button, .. }
                        if !gui_want_capture_mouse =>
                    {
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

        self.state.keyboard.update(keyboard_events, self.state.dt);
        self.state.mouse_state.update(&new_mouse_state);

        running
    }

    pub fn draw_forever(mut self, mut callback: impl FnMut(&FrameState) -> SnoozyRef<Texture>) {
        tracing::debug!("Rendertoy::draw_forever");

        let mut running = true;
        while running {
            let window_size_pixels = {
                let size = self
                    .window
                    .get_inner_size()
                    .map(|s| s.to_physical(self.window.get_hidpi_factor()))
                    .unwrap_or(winit::dpi::PhysicalSize::new(1.0, 1.0));
                (size.width as u32, size.height as u32)
            };

            let state = &mut self.state;
            let window = &self.window;
            let imgui = &mut self.imgui;
            let imgui_backend = &mut self.imgui_backend;
            let gui_placeholder_texture_view = self.gui_placeholder_tex.view;

            let render_result = self.renderer.render_frame(|renderer| {
                let vk_state = self::vulkan::vk_state();

                let final_texture =
                    state.draw_with_frame_snapshot(window_size_pixels, &mut callback);
                let cb = vk_state.current_frame().command_buffer.lock().unwrap();
                let cb = cb.cb;

                let currently_debugged_texture = state.get_currently_debugged_texture().clone();

                let gui_texture_view = if state.show_gui {
                    let ui = imgui_backend.prepare_frame(&window, imgui, state.dt);
                    {
                        state.dump_next_frame_dot_graph =
                            ui.button(im_str!("Dump frame.dot"), [0.0, 0.0]);
                        ui.spacing();

                        if ui
                            .collapsing_header(im_str!("GPU passes"))
                            .default_open(true)
                            .build()
                        {
                            if let Some(stats) = renderer.get_gpu_profiler_stats() {
                                state.selected_debug_name = RendertoyState::draw_profiling_stats(
                                    &ui,
                                    state.average_frame_time,
                                    state.time_to_first_frame,
                                    stats,
                                    &currently_debugged_texture,
                                );
                            }
                        }

                        crate::warnings::with_drain_warnings(|warnings| {
                            if !warnings.is_empty() {
                                if ui
                                    .collapsing_header(&im_str!(
                                        "Warnings ({})###warnings",
                                        warnings.len()
                                    ))
                                    .default_open(false)
                                    .build()
                                {
                                    warnings.sort();
                                    for warning in warnings.drain(..) {
                                        ui.text(warning);
                                    }
                                }
                            }
                        });
                    }

                    let gui_extent = vk_state.swapchain.as_ref().unwrap().surface_resolution;
                    imgui_backend
                        .render(&window, (gui_extent.width, gui_extent.height), ui, cb)
                        .expect("gui texture")
                } else {
                    gui_placeholder_texture_view
                };

                (final_texture, gui_texture_view)
            });

            if let RenderFrameStatus::SwapchainRecreated = render_result {
                imgui_backend.destroy_graphics_resources();
                imgui_backend.create_graphics_resources();
            }

            running = self.next_frame();
        }
    }
}

impl RendertoyState {
    fn get_currently_debugged_texture(&self) -> Option<String> {
        self.selected_debug_name
            .clone()
            .or(self.locked_debug_name.clone())
    }

    fn draw_with_frame_snapshot<F>(
        &mut self,
        window_size_pixels: (u32, u32),
        callback: &mut F,
    ) -> vk::ImageView
    where
        F: FnMut(&FrameState) -> SnoozyRef<Texture>,
    {
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
            tracing::info!(
                "CPU frame time: {:.2}ms ({:.2} fps)",
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

        let final_texture = {
            let tex = tex.clone();
            self.rt.block_on(async move {
                let snapshot = get_snapshot(move |f| {
                    tokio::task::spawn(async move {
                        f();
                    });
                });
                let final_texture: Texture = (*snapshot.get(tex).await).clone();
                final_texture.view
            })
        };

        if self.time_to_first_frame.is_none() {
            self.time_to_first_frame = Some(self.initialization_instant.elapsed());
        }

        if self.dump_next_frame_dot_graph {
            self.dump_next_frame_dot_graph = false;
            let dot = generate_dot_graph_from_snoozy_ref(
                tex.into(),
                Some(&["compute_tex", "raster_tex"]),
                &[],
                Some("rankdir = BT"),
            );

            use std::fs::File;
            use std::io::prelude::*;

            let mut file = File::create("frame.dot").expect("File::create");
            file.write_all(dot.as_bytes()).expect("file.write_all");
        }

        let debugged_texture = {
            let mut debugged_texture: Option<vk::ImageView> = None;
            gpu_debugger::with_textures(|data| {
                debugged_texture = self
                    .get_currently_debugged_texture()
                    .and_then(|name| data.textures.get(&name).cloned());
            });
            debugged_texture
        };

        debugged_texture.unwrap_or(final_texture)
    }

    fn draw_profiling_stats(
        ui: &imgui::Ui,
        average_frame_time: f32,
        time_to_first_frame: Option<std::time::Duration>,
        stats: &GpuProfilerStats,
        currently_debugged_texture: &Option<String>,
    ) -> Option<String> {
        let mut selected_name = None;
        if let Some(time_to_first_frame) = time_to_first_frame {
            ui.text(format!("Time to first frame: {:?}", time_to_first_frame));
        }

        ui.text(format!(
            "CPU frame time: {:.2}ms ({:.1} fps)",
            1000.0 * average_frame_time,
            1.0 / average_frame_time
        ));
        //let mut total_time_ms = 0.0;

        for (_scope_id, scope) in stats.scopes.iter() {
            let text = format!("{}: {:.3}ms", scope.name, scope.average_duration_millis());
            //let text = &scope.name;

            let style = if Some(&scope.name) == currently_debugged_texture.as_ref() {
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
                selected_name = Some(scope.name.clone());
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
}

fn generate_dot_graph_from_snoozy_ref(
    root: OpaqueSnoozyRef,
    ops_to_include: Option<&[&'static str]>,
    ops_to_skip: &[&'static str],
    dot_graph_attribs: Option<&'static str>,
) -> String {
    use snoozy::{OpaqueSnoozyRefInner, SnoozyRefDependency};

    let should_include_op = |op: &SnoozyRefDependency| {
        let op_name = op.recipe_info.read().unwrap().recipe_meta.op_name;

        !ops_to_skip.contains(&op_name)
            && ops_to_include
                .map(|names| names.contains(&op_name))
                .unwrap_or(true)
    };

    let get_node_name = |r: &OpaqueSnoozyRefInner| -> String {
        let info = r.recipe_info.read().unwrap();
        if let Some(ref build_record) = info.build_record {
            if let Some(ref debug_name) = build_record.build_result.debug_info.debug_name {
                return debug_name.clone();
            }
        }

        format!("{}", info.recipe_meta.op_name)
    };

    use petgraph::*;
    use std::collections::HashMap;

    let mut node_indices: HashMap<usize, _> = HashMap::new();
    let root_name = get_node_name(&root.inner);

    let mut graph = Graph::<String, String>::new();
    let root_idx = graph.add_node(root_name);
    node_indices.insert(root.get_transient_op_id(), root_idx);

    let mut stack: Vec<(SnoozyRefDependency, _)> = Vec::new();
    stack.push((SnoozyRefDependency(root.inner), root_idx));

    while let Some((r, r_idx)) = stack.pop() {
        let info = r.recipe_info.read().unwrap();

        if let Some(build_record) = info.build_record.as_ref() {
            for dep in build_record.dependencies.iter() {
                let dep_op_id = dep.get_transient_op_id();

                if let Some(dep_idx) = node_indices.get(&dep_op_id) {
                    graph.extend_with_edges(&[(r_idx, *dep_idx)]);
                } else if should_include_op(&dep) {
                    let dep_name = get_node_name(dep);
                    let dep_idx = graph.add_node(dep_name.clone());
                    node_indices.insert(dep_op_id, dep_idx);
                    graph.extend_with_edges(&[(r_idx, dep_idx)]);
                    stack.push((dep.clone(), dep_idx));
                }
            }
        }
    }

    let dot = crate::dot::Dot::new(&graph, dot_graph_attribs);
    format!("{}", dot)
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
