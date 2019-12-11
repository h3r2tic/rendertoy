use crate::gpu_debugger;
use crate::gpu_profiler::{self, GpuProfilerStats};
use crate::shader;
use crate::vulkan;
use ash::vk;
use std::sync::Arc;

pub struct Renderer {
    gpu_profiler_stats: Option<GpuProfilerStats>,
    swapchain_acquired_semaphore_idx: usize,
    present_descriptor_sets: Vec<vk::DescriptorSet>,
    present_pipeline: shader::ComputePipeline,
    window: Arc<winit::Window>,
}

pub enum RenderFrameResult {
    Ok,
    SwapchainLost,
    SwapchainRecreated,
}

impl Renderer {
    pub fn new(window: Arc<winit::Window>, graphics_debugging: bool, vsync: bool) -> Self {
        vulkan::initialize_vk_state(
            vulkan::VkKitchenSink::new(&window, graphics_debugging, vsync).unwrap(),
        );

        let (present_descriptor_sets, present_pipeline) =
            Self::create_present_descriptor_sets_and_pipeline();

        Self {
            gpu_profiler_stats: None,
            swapchain_acquired_semaphore_idx: 0,
            present_descriptor_sets,
            present_pipeline,
            window,
        }
    }

    pub fn render_frame(
        &mut self,
        mut callback: impl FnMut(&Self) -> (vk::ImageView, vk::ImageView),
    ) -> RenderFrameResult {
        use crate::vulkan::*;
        use ash::version::DeviceV1_0;

        if crate::vulkan::vk().swapchain.is_none() {
            return self.resize();
        }

        let present_index = {
            let vk = vk();
            let swapchain = vk.swapchain.as_ref().unwrap();

            self.swapchain_acquired_semaphore_idx =
                (self.swapchain_acquired_semaphore_idx + 1) % vk.frame_data.len();

            let present_index = unsafe {
                let (present_index, _) = {
                    match vk.swapchain_loader.acquire_next_image(
                        swapchain.swapchain,
                        std::u64::MAX,
                        swapchain.swapchain_acquired_semaphores
                            [self.swapchain_acquired_semaphore_idx],
                        vk::Fence::null(),
                    ) {
                        Ok(res) => res,
                        Err(err)
                            if err == vk::Result::ERROR_OUT_OF_DATE_KHR
                                || err == vk::Result::SUBOPTIMAL_KHR =>
                        {
                            return self.resize();
                        }
                        err @ _ => {
                            panic!("Could not acquire swapchain image: {:?}", err);
                        }
                    }
                };

                present_index as usize
            };

            present_index
        };

        with_vk_mut(|vk| vk.begin_frame(present_index));
        let current_frame_data_idx;

        let (wait_semaphore, signal_semaphore) = {
            let vk = vk();
            let swapchain = vk.swapchain.as_ref().unwrap();

            current_frame_data_idx = present_index;
            assert_eq!(vk.current_frame_data_idx, Some(current_frame_data_idx));

            (
                swapchain.swapchain_acquired_semaphores[self.swapchain_acquired_semaphore_idx],
                swapchain.rendering_complete_semaphores[current_frame_data_idx],
            )
        };

        record_submit_commandbuffer(
            &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            &[wait_semaphore],
            &[signal_semaphore],
            |vk| {
                let swapchain = vk.swapchain.as_ref().unwrap();
                let present_image = swapchain.present_images[current_frame_data_idx];

                record_image_barrier(
                    &vk.device,
                    vk.current_frame().command_buffer.lock().unwrap().cb,
                    ImageBarrier::new(
                        present_image,
                        vk_sync::AccessType::Present,
                        vk_sync::AccessType::ComputeShaderWrite,
                    )
                    .with_discard(true),
                );

                let (final_texture_view, gui_texture_view) = callback(self);

                let cb = vk.current_frame().command_buffer.lock().unwrap().cb;

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
                                    .image_view(
                                        swapchain.present_image_views[current_frame_data_idx],
                                    )
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
                    let output_size_pixels = vk.swapchain_size_pixels();
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

        let vk = vk();
        let swapchain = vk.swapchain.as_ref().unwrap();
        let wait_semaphores = [swapchain.rendering_complete_semaphores[current_frame_data_idx]];
        let swapchains = [swapchain.swapchain];
        let image_indices = [present_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            match vk
                .swapchain_loader
                .queue_present(vk.present_queue, &present_info)
            {
                Ok(_) => (),
                Err(err)
                    if err == vk::Result::ERROR_OUT_OF_DATE_KHR
                        || err == vk::Result::SUBOPTIMAL_KHR =>
                {
                    // Handled it in the next frame
                }
                err @ _ => {
                    panic!("Could not acquire swapchain image: {:?}", err);
                }
            }
        }

        gpu_profiler::end_frame();
        gpu_debugger::end_frame();

        self.gpu_profiler_stats = Some(gpu_profiler::get_stats());
        RenderFrameResult::Ok
    }

    pub fn get_gpu_profiler_stats(&self) -> Option<&GpuProfilerStats> {
        self.gpu_profiler_stats.as_ref()
    }

    fn resize(&mut self) -> RenderFrameResult {
        let logical_size = self.window.get_inner_size().unwrap();
        let dpi_factor = self.window.get_hidpi_factor();
        let phys_size = logical_size.to_physical(dpi_factor);

        if crate::vulkan::vk_resize(phys_size.width as u32, phys_size.height as u32) {
            RenderFrameResult::SwapchainRecreated
        } else {
            RenderFrameResult::SwapchainLost
        }
    }

    fn create_present_descriptor_sets_and_pipeline(
    ) -> (Vec<vk::DescriptorSet>, shader::ComputePipeline) {
        use crate::vulkan::*;
        use ash::version::DeviceV1_0;
        let vk = vk();

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

        (present_descriptor_sets, present_pipeline)
    }
}

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

        Ok(crate::shader::ComputePipeline {
            pipeline_layout,
            pipeline,
        })
    }
}
