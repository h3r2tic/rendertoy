use crate::gpu_debugger;
use crate::gpu_profiler::{self, GpuProfilerStats};
use crate::shader;
use crate::vulkan::*;
use ash::version::DeviceV1_0;
use ash::vk;
use std::sync::Arc;

pub struct Renderer {
    gpu_profiler_stats: Option<GpuProfilerStats>,
    present_descriptor_sets: Vec<vk::DescriptorSet>,
    present_pipeline: shader::ComputePipeline,
    window: Arc<winit::Window>,
}

pub enum RenderFrameStatus {
    Ok,
    SwapchainLost,
    SwapchainRecreated,
}

impl Renderer {
    pub fn new(
        window: Arc<winit::Window>,
        graphics_debugging: bool,
        vsync: bool,
        device_index: usize,
    ) -> Self {
        initialize_vulkan_backend(&window, graphics_debugging, vsync, device_index);

        let (present_descriptor_sets, present_pipeline) =
            Self::create_present_descriptor_sets_and_pipeline();

        Self {
            gpu_profiler_stats: None,
            present_descriptor_sets,
            present_pipeline,
            window,
        }
    }

    pub fn begin_setup_frame(&mut self) -> RenderFrameStatus {
        // The swapchain was lost -- possibly due to the window being minimized.
        // See if we can re-create it.
        if vk_state().swapchain.is_none() {
            return self.resize();
        }

        let fs = with_vk_state_mut(VkBackendState::begin_frame);
        let fs = match fs {
            Ok(s) => s,
            Err(BeginFrameErr::RecreateFramebuffer) => {
                return self.resize();
            }
        };

        crate::vulkan::begin_render_frame(
            &fs,
            |vk, _present_index, present_image, _present_image_view| {
                let cb = vk_state().current_frame().command_buffer.lock().unwrap().cb;
                record_image_barrier(
                    &vk.device,
                    cb,
                    ImageBarrier::new(
                        present_image,
                        vk_sync::AccessType::Nothing,
                        vk_sync::AccessType::Present,
                    ),
                );
            },
        );

        RenderFrameStatus::Ok
    }

    pub fn end_setup_frame(&mut self) -> RenderFrameStatus {
        let fs = vk_state().get_begin_frame_state();

        crate::vulkan::end_render_frame(&fs);
        vk_state().end_frame();

        gpu_profiler::end_frame();
        gpu_debugger::end_frame();

        self.gpu_profiler_stats = Some(gpu_profiler::get_stats());
        RenderFrameStatus::Ok
    }

    pub fn render_frame(
        &mut self,
        mut callback: impl FnMut(&Self) -> (vk::ImageView, vk::ImageView),
    ) -> RenderFrameStatus {
        // The swapchain was lost -- possibly due to the window being minimized.
        // See if we can re-create it.
        if vk_state().swapchain.is_none() {
            return self.resize();
        }

        let fs = with_vk_state_mut(VkBackendState::begin_frame);
        let fs = match fs {
            Ok(s) => s,
            Err(BeginFrameErr::RecreateFramebuffer) => {
                return self.resize();
            }
        };

        crate::vulkan::begin_render_frame(
            &fs,
            |vk, present_index, present_image, present_image_view| {
                record_image_barrier(
                    &vk.device,
                    vk_state().current_frame().command_buffer.lock().unwrap().cb,
                    ImageBarrier::new(
                        present_image,
                        vk_sync::AccessType::Present,
                        vk_sync::AccessType::ComputeShaderWrite,
                    )
                    .with_discard(true),
                );

                let (final_texture_view, gui_texture_view) = callback(self);

                let cb = vk_state().current_frame().command_buffer.lock().unwrap().cb;

                unsafe {
                    vk.device.update_descriptor_sets(
                        &[
                            vk::WriteDescriptorSet::builder()
                                .dst_set(self.present_descriptor_sets[present_index])
                                .dst_binding(0)
                                .dst_array_element(0)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .image_info(&[vk::DescriptorImageInfo::builder()
                                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                    .image_view(final_texture_view)
                                    .build()])
                                .build(),
                            vk::WriteDescriptorSet::builder()
                                .dst_set(self.present_descriptor_sets[present_index])
                                .dst_binding(1)
                                .dst_array_element(0)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .image_info(&[vk::DescriptorImageInfo::builder()
                                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                    .image_view(gui_texture_view)
                                    .build()])
                                .build(),
                            vk::WriteDescriptorSet::builder()
                                .dst_set(self.present_descriptor_sets[present_index])
                                .dst_binding(2)
                                .dst_array_element(0)
                                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                .image_info(&[vk::DescriptorImageInfo::builder()
                                    .image_layout(vk::ImageLayout::GENERAL)
                                    .image_view(present_image_view)
                                    .build()])
                                .build(),
                        ],
                        &[],
                    );

                    vk.device.cmd_bind_pipeline(
                        cb,
                        vk::PipelineBindPoint::COMPUTE,
                        self.present_pipeline.pipeline,
                    );
                    vk.device.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::COMPUTE,
                        self.present_pipeline.pipeline_layout,
                        0,
                        &[self.present_descriptor_sets[present_index]],
                        &[],
                    );
                    let output_size_pixels = vk_state().swapchain_size_pixels();
                    let push_constants: (f32, f32) = (
                        1.0 / output_size_pixels.0 as f32,
                        1.0 / output_size_pixels.1 as f32,
                    );
                    vk.device.cmd_push_constants(
                        cb,
                        self.present_pipeline.pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        std::slice::from_raw_parts(
                            &push_constants.0 as *const f32 as *const u8,
                            2 * 4,
                        ),
                    );
                    vk.device.cmd_dispatch(
                        cb,
                        (output_size_pixels.0 + 7) / 8,
                        (output_size_pixels.1 + 7) / 8,
                        1,
                    );
                }

                record_image_barrier(
                    &vk.device,
                    cb,
                    ImageBarrier::new(
                        present_image,
                        vk_sync::AccessType::ComputeShaderWrite,
                        vk_sync::AccessType::Present,
                    ),
                );
            },
        );

        crate::vulkan::end_render_frame(&fs);

        vk_state().end_frame();

        gpu_profiler::end_frame();
        gpu_debugger::end_frame();

        self.gpu_profiler_stats = Some(gpu_profiler::get_stats());
        RenderFrameStatus::Ok
    }

    pub fn get_gpu_profiler_stats(&self) -> Option<&GpuProfilerStats> {
        self.gpu_profiler_stats.as_ref()
    }

    fn resize(&mut self) -> RenderFrameStatus {
        let logical_size = self.window.get_inner_size().unwrap();
        let dpi_factor = self.window.get_hidpi_factor();
        let phys_size = logical_size.to_physical(dpi_factor);

        if vk_resize(phys_size.width as u32, phys_size.height as u32) {
            RenderFrameStatus::SwapchainRecreated
        } else {
            RenderFrameStatus::SwapchainLost
        }
    }

    fn create_present_descriptor_sets_and_pipeline(
    ) -> (Vec<vk::DescriptorSet>, shader::ComputePipeline) {
        let (vk, vk_state) = vk_all();

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
                                .immutable_samplers(&[vk.samplers[SAMPLER_LINEAR]])
                                .build(),
                        ])
                        .build(),
                    None,
                )
                .unwrap()
        };

        let present_descriptor_sets =
            vk_state.create_present_descriptor_sets(present_descriptor_set_layout);
        let present_pipeline =
            create_present_compute_pipeline(&vk.device, present_descriptor_set_layout)
                .expect("create_present_compute_pipeline");

        (present_descriptor_sets, present_pipeline)
    }
}

fn create_present_compute_pipeline(
    vk_device: &ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> snoozy::Result<crate::shader::ComputePipeline> {
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

        Ok(crate::shader::ComputePipeline {
            pipeline_layout,
            pipeline,
        })
    }
}
