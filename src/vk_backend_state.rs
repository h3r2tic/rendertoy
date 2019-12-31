//use ash::extensions::nv::RayTracing;
use crate::gpu_profiler::GpuProfilerQueryId;
use crate::vk_render_device::*;
use crate::vulkan::{vk, vk_add_setup_command, vk_all, with_vk_state_mut};
use ash::extensions::khr::{Surface, Swapchain};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::{vk, Device};
use std::error::Error;
use std::sync::Mutex;

pub struct VkCommandBufferData {
    pub(crate) cb: vk::CommandBuffer,
    pool: vk::CommandPool,
}

impl Drop for VkCommandBufferData {
    fn drop(&mut self) {
        unsafe {
            vk().device.destroy_command_pool(self.pool, None);
        }
    }
}

pub struct LinearUniformBuffer {
    write_head: std::sync::atomic::AtomicUsize,
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    allocation_info: vk_mem::AllocationInfo,
    size: vk::DeviceSize,
    mapped_ptr: *mut u8,
    min_offset_alignment: usize,
}

unsafe impl Send for LinearUniformBuffer {}
unsafe impl Sync for LinearUniformBuffer {}

impl Drop for LinearUniformBuffer {
    fn drop(&mut self) {
        let vk = vk();
        self.unmap(&vk.device, &vk.allocator);
        vk.allocator
            .destroy_buffer(self.buffer, &self.allocation)
            .unwrap()
    }
}

impl LinearUniformBuffer {
    fn new(
        size: vk::DeviceSize,
        allocator: &vk_mem::Allocator,
        min_offset_alignment: usize,
    ) -> Self {
        let usage: vk::BufferUsageFlags = vk::BufferUsageFlags::UNIFORM_BUFFER;

        let mem_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::CpuToGpu,
            ..Default::default()
        };

        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        let (buffer, allocation, allocation_info) =
            allocator.create_buffer(&buffer_info, &mem_info).unwrap();

        let mapped_ptr: *mut u8 = allocator.map_memory(&allocation).expect("map_memory");

        Self {
            write_head: std::sync::atomic::AtomicUsize::new(0),
            buffer,
            allocation,
            allocation_info,
            size,
            mapped_ptr,
            min_offset_alignment,
        }
    }

    pub fn map(&mut self, allocator: &vk_mem::Allocator) {
        if self.mapped_ptr == std::ptr::null_mut() {
            self.mapped_ptr = allocator.map_memory(&self.allocation).unwrap();

            assert!(
                self.mapped_ptr != std::ptr::null_mut(),
                "failed to map uniform buffer"
            );
        }
    }

    pub fn unmap(&mut self, device: &Device, allocator: &vk_mem::Allocator) {
        if self.mapped_ptr != std::ptr::null_mut() {
            let bytes_written = self
                .write_head
                .swap(0, std::sync::atomic::Ordering::Relaxed);
            let alignment = 256; // TODO
            let bytes_written_pad = (bytes_written + alignment - 1) & !(alignment - 1);

            let mapped_ranges = [vk::MappedMemoryRange {
                memory: self.allocation_info.get_device_memory(),
                offset: self.allocation_info.get_offset() as vk::DeviceSize,
                size: bytes_written_pad as vk::DeviceSize,
                ..Default::default()
            }];

            unsafe { device.flush_mapped_memory_ranges(&mapped_ranges) }.unwrap();

            allocator
                .unmap_memory(&self.allocation)
                .expect("unmap_memory");
            self.mapped_ptr = std::ptr::null_mut();
        }
    }

    pub fn allocate(&self, bytes_count: usize) -> snoozy::Result<(vk::Buffer, u64, &mut [u8])> {
        assert!(self.mapped_ptr != std::ptr::null_mut());

        unsafe {
            let alloc_size =
                (bytes_count + self.min_offset_alignment - 1) & self.min_offset_alignment;
            let start_offset = self
                .write_head
                .fetch_add(alloc_size, std::sync::atomic::Ordering::Relaxed);

            if start_offset + bytes_count <= self.size as usize {
                Ok((
                    self.buffer,
                    start_offset as u64,
                    std::slice::from_raw_parts_mut(
                        self.mapped_ptr.offset(start_offset as isize),
                        bytes_count,
                    ),
                ))
            } else {
                bail!("Out of memory in LinearUniformBuffer::allocate")
            }
        }
    }
}

pub struct VkProfilerData {
    pub query_pool: vk::QueryPool,
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    next_query_id: std::sync::atomic::AtomicU32,
    gpu_profiler_query_ids: Vec<std::cell::Cell<GpuProfilerQueryId>>,
}

impl Drop for VkProfilerData {
    fn drop(&mut self) {
        unsafe {
            let vk = vk();
            vk.allocator
                .destroy_buffer(self.buffer, &self.allocation)
                .unwrap();

            vk.device.destroy_query_pool(self.query_pool, None);
        }

        let valid_query_count = self
            .next_query_id
            .load(std::sync::atomic::Ordering::Relaxed) as usize;

        crate::gpu_profiler::forget_queries(
            self.gpu_profiler_query_ids[0..valid_query_count]
                .iter()
                .map(std::cell::Cell::take),
        );
    }
}

const MAX_QUERY_COUNT: usize = 1024;

impl VkProfilerData {
    fn new(device: &Device, allocator: &vk_mem::Allocator) -> Self {
        let usage: vk::BufferUsageFlags = vk::BufferUsageFlags::TRANSFER_DST;

        let mem_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuToCpu,
            ..Default::default()
        };

        let (buffer, allocation, _allocation_info) = {
            let buffer_info = vk::BufferCreateInfo::builder()
                .size(MAX_QUERY_COUNT as u64 * 8 * 2)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .build();

            allocator
                .create_buffer(&buffer_info, &mem_info)
                .expect("vma::create_buffer")
        };

        let pool_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(MAX_QUERY_COUNT as u32 * 2);

        Self {
            query_pool: unsafe { device.create_query_pool(&pool_info, None) }
                .expect("create_query_pool"),
            buffer,
            allocation,
            next_query_id: Default::default(),
            gpu_profiler_query_ids: vec![
                std::cell::Cell::new(GpuProfilerQueryId::default());
                MAX_QUERY_COUNT
            ],
        }
    }

    pub fn get_query_id(&self, gpu_profiler_query_id: GpuProfilerQueryId) -> u32 {
        // TODO: handle running out of queries
        let id = self
            .next_query_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.gpu_profiler_query_ids[id as usize].set(gpu_profiler_query_id);
        id
    }

    // Two timing values per query
    fn retrieve_previous_result(
        &self,
        allocator: &vk_mem::Allocator,
    ) -> (Vec<GpuProfilerQueryId>, Vec<u64>) {
        let valid_query_count = self
            .next_query_id
            .load(std::sync::atomic::Ordering::Relaxed) as usize;

        let mapped_ptr = allocator
            .map_memory(&self.allocation)
            .expect("mapping a query buffer failed") as *mut u64;

        let result =
            unsafe { std::slice::from_raw_parts(mapped_ptr, valid_query_count * 2) }.to_owned();

        allocator
            .unmap_memory(&self.allocation)
            .expect("unmapping a query buffer failed");

        (
            self.gpu_profiler_query_ids[0..valid_query_count]
                .iter()
                .map(std::cell::Cell::get)
                .collect(),
            result,
        )
    }

    fn begin_frame(&self, device: &Device, cmd: vk::CommandBuffer) {
        unsafe {
            device.cmd_reset_query_pool(cmd, self.query_pool, 0, MAX_QUERY_COUNT as u32 * 2);
        }

        self.next_query_id
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    fn finish_frame(&self, device: &Device, cmd: vk::CommandBuffer) {
        let valid_query_count = self
            .next_query_id
            .load(std::sync::atomic::Ordering::Relaxed);

        unsafe {
            device.cmd_copy_query_pool_results(
                cmd,
                self.query_pool,
                0,
                valid_query_count * 2,
                self.buffer,
                0,
                8,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            );
        }
    }
}

pub struct VkFrameData {
    pub uniforms: LinearUniformBuffer,
    pub descriptor_pool: Mutex<vk::DescriptorPool>,
    pub command_buffer: Mutex<VkCommandBufferData>,
    pub submit_done_fence: vk::Fence,
    pub profiler_data: VkProfilerData,
    pub frame_cleanup: Mutex<Vec<Box<dyn Fn(&VkRenderDevice) + Send + Sync>>>,
}

impl Drop for VkFrameData {
    fn drop(&mut self) {
        let vk = vk();
        unsafe {
            vk.device.destroy_descriptor_pool(
                std::mem::replace(&mut self.descriptor_pool, Default::default())
                    .into_inner()
                    .unwrap(),
                None,
            );

            vk.device.destroy_fence(self.submit_done_fence, None);
        }
    }
}

pub const SAMPLER_LINEAR: usize = 0;

pub struct VkBackendState {
    pub swapchain: Option<VkSwapchain>,
    pub(crate) swapchain_create_info: VkSwapchainCreateInfo,
    pub(crate) swapchain_acquired_semaphore_idx: usize,

    pub bindless_buffers_descriptor_set: vk::DescriptorSet,
    pub bindless_images_descriptor_set: vk::DescriptorSet,

    // TODO: Those also guard updates to the descriptor sets
    pub bindless_buffers_next_descriptor: Mutex<u32>,
    pub bindless_images_next_descriptor: Mutex<u32>,

    pub frame_data: Vec<VkFrameData>,
    pub current_frame_data_idx: Option<usize>,

    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
}

pub struct ImageBarrier {
    image: vk::Image,
    prev_access: vk_sync::AccessType,
    next_access: vk_sync::AccessType,
    discard: bool,
}

impl ImageBarrier {
    pub fn new(
        image: vk::Image,
        prev_access: vk_sync::AccessType,
        next_access: vk_sync::AccessType,
    ) -> Self {
        Self {
            image,
            prev_access,
            next_access,
            discard: false,
        }
    }

    pub fn with_discard(mut self, discard: bool) -> Self {
        self.discard = discard;
        self
    }
}

fn allocate_frame_descriptor_pool(device: &Device) -> vk::DescriptorPool {
    let descriptor_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: 1 << 20,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLER,
            descriptor_count: 4,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1 << 20,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: 1 << 20,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1 << 20,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            descriptor_count: 1 << 20,
        },
    ];

    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&descriptor_sizes)
        .max_sets(1 << 16);

    unsafe {
        device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap()
    }
}

fn allocate_frame_command_buffer(
    device: &Device,
    present_queue_family_index: u32,
) -> VkCommandBufferData {
    let pool_create_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(present_queue_family_index);

    let pool = unsafe { device.create_command_pool(&pool_create_info, None).unwrap() };

    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(1)
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    let cb = unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .unwrap()
    }[0];

    VkCommandBufferData { cb, pool }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct VkSwapchainCreateInfo {
    pub(crate) surface_format: vk::SurfaceFormatKHR,
    pub(crate) surface_resolution: vk::Extent2D,
    pub(crate) vsync: bool,
}

#[derive(Default)]
pub struct VkSwapchain {
    pub swapchain: vk::SwapchainKHR,
    pub present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,
    pub surface_resolution: vk::Extent2D,
    pub surface_format: vk::SurfaceFormatKHR,
    pub swapchain_acquired_semaphores: Vec<vk::Semaphore>,
    pub rendering_complete_semaphores: Vec<vk::Semaphore>,
}

impl Drop for VkSwapchain {
    fn drop(&mut self) {
        unsafe {
            let vk = vk();
            for img in self.present_image_views.drain(..) {
                vk.device.destroy_image_view(img, None);
            }

            // TODO: semaphores

            vk.swapchain_loader.destroy_swapchain(self.swapchain, None);
        }
    }
}

// TODO: Result
pub(crate) fn create_swapchain(
    device: &Device,
    pdevice: vk::PhysicalDevice,
    swapchain_loader: &Swapchain,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
    info: VkSwapchainCreateInfo,
) -> Option<VkSwapchain> {
    let surface_capabilities =
        unsafe { surface_loader.get_physical_device_surface_capabilities(pdevice, surface) }
            .unwrap();
    let mut desired_image_count = surface_capabilities.min_image_count + 1;
    if surface_capabilities.max_image_count > 0
        && desired_image_count > surface_capabilities.max_image_count
    {
        desired_image_count = surface_capabilities.max_image_count;
    }

    //dbg!(&surface_capabilities);
    let surface_resolution = match surface_capabilities.current_extent.width {
        std::u32::MAX => info.surface_resolution,
        _ => surface_capabilities.current_extent,
    };

    if 0 == surface_resolution.width || 0 == surface_resolution.height {
        return None;
    }

    let present_modes =
        unsafe { surface_loader.get_physical_device_surface_present_modes(pdevice, surface) }
            .unwrap();
    let desired_present_mode = if info.vsync {
        vk::PresentModeKHR::FIFO_RELAXED
    } else {
        vk::PresentModeKHR::MAILBOX
    };
    let present_mode = present_modes
        .iter()
        .cloned()
        .find(|&mode| mode == desired_present_mode)
        .unwrap_or(vk::PresentModeKHR::FIFO);

    let pre_transform = if surface_capabilities
        .supported_transforms
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        surface_capabilities.current_transform
    };

    let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(desired_image_count)
        .image_color_space(info.surface_format.color_space)
        .image_format(info.surface_format.format)
        .image_extent(surface_resolution)
        .image_usage(vk::ImageUsageFlags::STORAGE)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(pre_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .image_array_layers(1)
        .build();

    let swapchain =
        unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }.unwrap();

    let present_images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }.unwrap();
    let present_image_views: Vec<vk::ImageView> = present_images
        .iter()
        .map(|&image| {
            let create_view_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(info.surface_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(image);
            unsafe { device.create_image_view(&create_view_info, None) }.unwrap()
        })
        .collect();

    let semaphore_create_info = vk::SemaphoreCreateInfo::default();
    let create_semaphores = || -> Vec<vk::Semaphore> {
        (0..present_images.len())
            .map(|_| unsafe { device.create_semaphore(&semaphore_create_info, None) }.unwrap())
            .collect()
    };

    let swapchain_acquired_semaphores = create_semaphores();
    let rendering_complete_semaphores = create_semaphores();

    Some(VkSwapchain {
        swapchain,
        present_images,
        present_image_views,
        surface_resolution,
        surface_format: info.surface_format,
        swapchain_acquired_semaphores,
        rendering_complete_semaphores,
    })
}

impl VkBackendState {
    pub(crate) fn new(
        render_device: &VkRenderDevice,
        window: &winit::Window,
        _graphics_debugging: bool,
        vsync: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let device = &render_device.device;
        let surface_loader = &render_device.surface_loader;
        let swapchain_loader = &render_device.swapchain_loader;
        let pdevice = render_device.pdevice;
        let surface = render_device.surface;
        let surface_format = render_device.surface_format;

        let allocator = &render_device.allocator;

        let physical_dimensions = window
            .get_inner_size()
            .unwrap()
            .to_physical(window.get_hidpi_factor());

        unsafe {
            let swapchain_create_info = VkSwapchainCreateInfo {
                surface_format: surface_format,
                surface_resolution: vk::Extent2D {
                    width: physical_dimensions.width as u32,
                    height: physical_dimensions.height as u32,
                },
                vsync,
            };
            let swapchain = create_swapchain(
                device,
                pdevice,
                &swapchain_loader,
                &surface_loader,
                surface,
                swapchain_create_info,
            )
            .unwrap();

            let bindless_buffers_descriptor_set =
                VkRenderDevice::create_bindless_resource_descriptor_set(
                    device,
                    vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                );
            let bindless_images_descriptor_set =
                VkRenderDevice::create_bindless_resource_descriptor_set(
                    device,
                    vk::DescriptorType::SAMPLED_IMAGE,
                );

            let depth_image_create_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT)
                .extent(vk::Extent3D {
                    width: swapchain.surface_resolution.width,
                    height: swapchain.surface_resolution.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let (depth_image, _allocation, _allocation_info) = allocator
                .create_image(
                    &depth_image_create_info,
                    &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::GpuOnly,
                        ..Default::default()
                    },
                )
                .unwrap();

            let depth_image_view_info = vk::ImageViewCreateInfo::builder()
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1)
                        .build(),
                )
                .image(depth_image)
                .format(depth_image_create_info.format)
                .view_type(vk::ImageViewType::TYPE_2D);

            let depth_image_view = device
                .create_image_view(&depth_image_view_info, None)
                .unwrap();

            vk_add_setup_command(move |vk, vk_frame| {
                let cb = vk_frame.command_buffer.lock().unwrap();
                let cb = cb.cb;

                record_image_aspect_barrier(
                    &vk.device,
                    cb,
                    vk::ImageAspectFlags::DEPTH,
                    ImageBarrier::new(
                        depth_image,
                        vk_sync::AccessType::Nothing,
                        vk_sync::AccessType::DepthAttachmentWriteStencilReadOnly,
                    )
                    .with_discard(true),
                )
            });

            let mut res = Self {
                swapchain: Some(swapchain),
                swapchain_acquired_semaphore_idx: 0,
                swapchain_create_info,
                frame_data: Vec::new(),
                current_frame_data_idx: None,
                bindless_buffers_descriptor_set,
                bindless_buffers_next_descriptor: Mutex::new(0),
                bindless_images_descriptor_set,
                bindless_images_next_descriptor: Mutex::new(0),
                depth_image,
                depth_image_view,
            };

            res.create_frame_data(render_device);
            Ok(res)
        }
    }

    pub fn begin_frame(&mut self) -> std::result::Result<BeginFrameState, BeginFrameErr> {
        let present_index = match self.acquire_next_image() {
            Ok(idx) => idx,
            Err(status) => return Err(status),
        } % self.frame_data.len();

        self.current_frame_data_idx = Some(present_index);

        let (wait_semaphore, signal_semaphore) = {
            let swapchain = self.swapchain.as_ref().unwrap();

            (
                swapchain.swapchain_acquired_semaphores[self.swapchain_acquired_semaphore_idx],
                swapchain.rendering_complete_semaphores[present_index],
            )
        };

        Ok(BeginFrameState {
            present_index,
            wait_semaphore,
            signal_semaphore,
        })
    }

    pub fn end_frame(&self, begin_frame_state: BeginFrameState) {
        let swapchain = self.swapchain.as_ref().unwrap();
        let wait_semaphores =
            [swapchain.rendering_complete_semaphores[begin_frame_state.present_index]];
        let swapchains = [swapchain.swapchain];
        let image_indices = [begin_frame_state.present_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let vk = vk();
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
    }

    pub fn current_frame<'a>(&'a self) -> &'a VkFrameData {
        &self.frame_data[self
            .current_frame_data_idx
            .expect("Rendering not started yet. `current_frame` not available")]
    }

    pub fn map_uniforms(&mut self) {
        let vk_frame = &mut self.frame_data[self
            .current_frame_data_idx
            .expect("Rendering not started yet. `current_frame` not available")];
        let vk = vk();
        vk_frame.uniforms.map(&vk.allocator);
    }

    pub fn unmap_uniforms(&mut self) {
        let vk_frame = &mut self.frame_data[self
            .current_frame_data_idx
            .expect("Rendering not started yet. `current_frame` not available")];
        let vk = vk();
        vk_frame.uniforms.unmap(&vk.device, &vk.allocator);
    }

    pub fn swapchain_size_pixels(&self) -> (u32, u32) {
        let vk::Extent2D { width, height } = self.swapchain.as_ref().unwrap().surface_resolution;
        (width, height)
    }

    pub(crate) fn create_present_descriptor_sets(
        &self,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Vec<vk::DescriptorSet> {
        unsafe {
            let descriptor_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: (self.frame_data.len() as u32) * 2,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: self.frame_data.len() as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLER,
                    descriptor_count: self.frame_data.len() as u32,
                },
            ];

            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&descriptor_sizes)
                .max_sets(self.frame_data.len() as u32);

            let vk = vk();
            let descriptor_pool = vk
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap();

            let descriptor_sets = vk
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&vec![descriptor_set_layout; self.frame_data.len()])
                        .build(),
                )
                .unwrap();

            descriptor_sets
        }
    }

    pub(crate) fn create_frame_data(&mut self, vk: &VkRenderDevice) {
        self.frame_data = (0..self.swapchain.as_ref().unwrap().present_images.len())
            .map(|_| {
                let uniforms = LinearUniformBuffer::new(
                    1 << 20,
                    &vk.allocator,
                    vk.device_properties
                        .limits
                        .min_uniform_buffer_offset_alignment as usize,
                );

                let submit_done_fence = unsafe {
                    vk.device.create_fence(
                        &vk::FenceCreateInfo::builder()
                            .flags(vk::FenceCreateFlags::SIGNALED)
                            .build(),
                        None,
                    )
                }
                .expect("Create fence failed.");

                let profiler_data = VkProfilerData::new(&vk.device, &vk.allocator);

                VkFrameData {
                    uniforms,
                    descriptor_pool: Mutex::new(allocate_frame_descriptor_pool(&vk.device)),
                    command_buffer: Mutex::new(allocate_frame_command_buffer(
                        &vk.device,
                        vk.present_queue_family_index,
                    )),
                    submit_done_fence,
                    profiler_data,
                    frame_cleanup: Mutex::new(Default::default()),
                }
            })
            .collect();
    }

    pub(crate) fn register_image_bindless_index(&self, view: vk::ImageView) -> u32 {
        let mut next = self.bindless_images_next_descriptor.lock().unwrap();

        let idx = *next;
        *next += 1;

        let vk = vk();
        unsafe {
            vk.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(self.bindless_images_descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(idx)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&[vk::DescriptorImageInfo::builder()
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(view)
                        .build()])
                    .build()],
                &[],
            )
        }

        idx
    }

    pub(crate) fn register_buffer_bindless_index(&self, view: vk::BufferView) -> u32 {
        let mut next = self.bindless_buffers_next_descriptor.lock().unwrap();

        let idx = *next;
        *next += 1;

        let vk = vk();
        unsafe {
            vk.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(self.bindless_buffers_descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(idx)
                    .descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                    .texel_buffer_view(&[view])
                    .build()],
                &[],
            )
        }

        idx
    }

    fn acquire_next_image(&mut self) -> std::result::Result<usize, BeginFrameErr> {
        let swapchain = self.swapchain.as_ref().unwrap();

        self.swapchain_acquired_semaphore_idx =
            (self.swapchain_acquired_semaphore_idx + 1) % self.frame_data.len();

        let present_index = unsafe {
            vk().swapchain_loader.acquire_next_image(
                swapchain.swapchain,
                std::u64::MAX,
                swapchain.swapchain_acquired_semaphores[self.swapchain_acquired_semaphore_idx],
                vk::Fence::null(),
            )
        }
        .map(|(val, _)| val as usize);

        match present_index {
            Ok(res) => Ok(res),
            Err(err)
                if err == vk::Result::ERROR_OUT_OF_DATE_KHR
                    || err == vk::Result::SUBOPTIMAL_KHR =>
            {
                Err(BeginFrameErr::RecreateFramebuffer)
            }
            err @ _ => {
                panic!("Could not acquire swapchain image: {:?}", err);
            }
        }
    }
}

pub struct BeginFrameState {
    pub present_index: usize,
    wait_semaphore: vk::Semaphore,
    signal_semaphore: vk::Semaphore,
}

pub enum BeginFrameErr {
    RecreateFramebuffer,
}

pub fn record_image_barrier(device: &Device, cb: vk::CommandBuffer, barrier: ImageBarrier) {
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    vk_sync::cmd::pipeline_barrier(
        device.fp_v1_0(),
        cb,
        None,
        &[],
        &[vk_sync::ImageBarrier {
            previous_accesses: &[barrier.prev_access],
            next_accesses: &[barrier.next_access],
            previous_layout: vk_sync::ImageLayout::Optimal,
            next_layout: vk_sync::ImageLayout::Optimal,
            discard_contents: barrier.discard,
            src_queue_family_index: 0,
            dst_queue_family_index: 0,
            image: barrier.image,
            range,
        }],
    );
}

pub fn record_image_aspect_barrier(
    device: &Device,
    cb: vk::CommandBuffer,
    aspect_mask: vk::ImageAspectFlags,
    barrier: ImageBarrier,
) {
    let range = vk::ImageSubresourceRange {
        aspect_mask,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    vk_sync::cmd::pipeline_barrier(
        device.fp_v1_0(),
        cb,
        None,
        &[],
        &[vk_sync::ImageBarrier {
            previous_accesses: &[barrier.prev_access],
            next_accesses: &[barrier.next_access],
            previous_layout: vk_sync::ImageLayout::Optimal,
            next_layout: vk_sync::ImageLayout::Optimal,
            discard_contents: barrier.discard,
            src_queue_family_index: 0,
            dst_queue_family_index: 0,
            image: barrier.image,
            range,
        }],
    );
}

impl Drop for VkBackendState {
    fn drop(&mut self) {
        unsafe {
            let vk = vk();
            vk.device.device_wait_idle().unwrap();
            // TODO:
            /*for s in self.swapchain_acquired_semaphores.iter() {
                self.device.destroy_semaphore(*s, None);
            }*/
            /*for s in self.frame_data.iter() {
                self.device
                    .destroy_semaphore(s.rendering_complete_semaphore, None);
            }*/
            /*self.swapchain_loader
            .destroy_swapchain(self.swapchain.swapchain, None);*/
            vk.device.destroy_device(None);
            vk.surface_loader.destroy_surface(vk.surface, None);
            if let Some(debug_report_loader) = vk.debug_report_loader.as_ref() {
                debug_report_loader
                    .destroy_debug_report_callback(vk.debug_call_back.unwrap(), None);
            }
            vk.instance.destroy_instance(None);
        }
    }
}

pub fn render_frame<F: FnOnce(&VkRenderDevice, usize, vk::Image, vk::ImageView)>(
    begin_frame_state: &BeginFrameState,
    render_fn: F,
) {
    let wait_mask: &[vk::PipelineStageFlags] = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
    let wait_semaphores: &[vk::Semaphore] = &[begin_frame_state.wait_semaphore];
    let signal_semaphores: &[vk::Semaphore] = &[begin_frame_state.signal_semaphore];

    unsafe {
        with_vk_state_mut(|vk_state| {
            let vk = vk();
            let submit_done_fence = vk_state.current_frame().submit_done_fence;
            vk.device
                .wait_for_fences(&[submit_done_fence], true, std::u64::MAX)
                .expect("Wait for fence failed.");

            let (query_ids, timing_pairs) = vk_state
                .current_frame()
                .profiler_data
                .retrieve_previous_result(&vk.allocator);

            let ns_per_tick = vk.device_properties.limits.timestamp_period;

            crate::gpu_profiler::report_durations_ticks(
                ns_per_tick,
                timing_pairs.chunks_exact(2).enumerate().map(
                    |(pair_idx, chunk)| -> (GpuProfilerQueryId, u64) {
                        (query_ids[pair_idx], chunk[1] - chunk[0])
                    },
                ),
            );

            vk_state.map_uniforms();
        });

        {
            let (vk, vk_state) = vk_all();
            let vk_frame = vk_state.current_frame();

            {
                {
                    let cb = vk_frame.command_buffer.lock().unwrap();

                    vk.device
                        .reset_command_buffer(cb.cb, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
                        .expect("Reset command buffer failed.");
                }

                for f in vk_frame.frame_cleanup.lock().unwrap().drain(..) {
                    (f)(vk);
                }

                {
                    let pool = vk_frame.descriptor_pool.lock().unwrap();
                    vk.device
                        .reset_descriptor_pool(*pool, Default::default())
                        .expect("reset_descriptor_pool");
                    drop(pool);
                }

                let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

                {
                    let cb = vk_frame.command_buffer.lock().unwrap();
                    let cb = cb.cb;
                    vk.device
                        .begin_command_buffer(cb, &command_buffer_begin_info)
                        .expect("Begin commandbuffer");

                    vk_frame.profiler_data.begin_frame(&vk.device, cb);
                }
            }

            for f in VK_SETUP_COMMANDS.lock().unwrap().drain(..).into_iter() {
                f(vk, vk_frame);
            }

            let swapchain = vk_state.swapchain.as_ref().unwrap();
            render_fn(
                vk,
                begin_frame_state.present_index,
                swapchain.present_images[begin_frame_state.present_index],
                swapchain.present_image_views[begin_frame_state.present_index],
            );
        }

        with_vk_state_mut(|vk| {
            vk.unmap_uniforms();
        });

        let (vk, vk_state) = vk_all();
        let vk_frame = vk_state.current_frame();

        {
            let cb = vk_frame.command_buffer.lock().unwrap();
            let cb = cb.cb;

            vk_frame.profiler_data.finish_frame(&vk.device, cb);

            vk.device.end_command_buffer(cb).expect("End commandbuffer");

            let submit_fence = vk_frame.submit_done_fence;
            vk.device
                .reset_fences(&[submit_fence])
                .expect("reset_fences");

            let command_buffers = vec![cb];

            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(wait_mask)
                .command_buffers(&command_buffers)
                .signal_semaphores(signal_semaphores);

            vk.device
                .queue_submit(vk.present_queue, &[submit_info.build()], submit_fence)
                .expect("queue submit failed.");
        }
    }
}

lazy_static! {
    pub(crate) static ref VK_SETUP_COMMANDS: Mutex<Vec<Box<dyn FnOnce(&VkRenderDevice, &VkFrameData) + Send + 'static>>> =
        { Mutex::new(Vec::new()) };
}
