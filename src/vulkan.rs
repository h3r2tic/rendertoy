//use ash::extensions::nv::RayTracing;
use crate::gpu_profiler::GpuProfilerQueryId;
use ash::extensions::{
    ext::DebugReport,
    khr::{Surface, Swapchain},
};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0, InstanceV1_1};
use ash::{vk, Device, Entry, Instance};
use std::error::Error;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::sync::Mutex;

unsafe extern "system" fn vulkan_debug_callback(
    _: vk::DebugReportFlagsEXT,
    _: vk::DebugReportObjectTypeEXT,
    _: u64,
    _: usize,
    _: i32,
    _: *const c_char,
    p_message: *const c_char,
    _: *mut c_void,
) -> u32 {
    tracing::error!("{:?}", CStr::from_ptr(p_message));
    vk::FALSE
}

fn extension_names(graphics_debugging: bool) -> Vec<*const i8> {
    let mut names = vec![vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr()];

    if graphics_debugging {
        names.push(DebugReport::name().as_ptr());
    }

    names
}

pub struct VkCommandBufferData {
    pub(crate) cb: vk::CommandBuffer,
    pool: vk::CommandPool,
}

impl Drop for VkCommandBufferData {
    fn drop(&mut self) {
        unsafe {
            vk_device().destroy_command_pool(self.pool, None);
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
        self.unmap();

        unsafe {
            let vk = vk_all();
            vk.allocator
                .destroy_buffer(self.buffer, &self.allocation)
                .unwrap()
        }
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

    pub fn map(&mut self) {
        if self.mapped_ptr == std::ptr::null_mut() {
            unsafe {
                self.mapped_ptr = vk_all().allocator.map_memory(&self.allocation).unwrap();
            }

            assert!(
                self.mapped_ptr != std::ptr::null_mut(),
                "failed to map uniform buffer"
            );
        }
    }

    pub fn unmap(&mut self) {
        if self.mapped_ptr != std::ptr::null_mut() {
            let bytes_written = self
                .write_head
                .swap(0, std::sync::atomic::Ordering::Relaxed);

            let mapped_ranges = [vk::MappedMemoryRange {
                memory: self.allocation_info.get_device_memory(),
                offset: self.allocation_info.get_offset() as vk::DeviceSize,
                size: bytes_written as vk::DeviceSize,
                ..Default::default()
            }];
            unsafe { vk_device().flush_mapped_memory_ranges(&mapped_ranges) }.unwrap();

            unsafe {
                vk_all()
                    .allocator
                    .unmap_memory(&self.allocation)
                    .expect("unmap_memory");
            }
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
            let vk = vk_all();
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
    fn retrieve_previous_result(&self) -> (Vec<GpuProfilerQueryId>, Vec<u64>) {
        let valid_query_count = self
            .next_query_id
            .load(std::sync::atomic::Ordering::Relaxed) as usize;

        let mapped_ptr = unsafe { vk_all() }
            .allocator
            .map_memory(&self.allocation)
            .expect("mapping a query buffer failed") as *mut u64;

        let result =
            unsafe { std::slice::from_raw_parts(mapped_ptr, valid_query_count * 2) }.to_owned();

        unsafe { vk_all() }
            .allocator
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
    pub frame_cleanup: Mutex<Vec<Box<dyn Fn(&VkKitchenSink) + Send + Sync>>>,
}

impl Drop for VkFrameData {
    fn drop(&mut self) {
        let device = vk_device();
        unsafe {
            device.destroy_descriptor_pool(
                std::mem::replace(&mut self.descriptor_pool, Default::default())
                    .into_inner()
                    .unwrap(),
                None,
            );

            device.destroy_fence(self.submit_done_fence, None);
        }
    }
}

pub const SAMPLER_LINEAR: usize = 0;

pub struct VkKitchenSink {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub device_properties: vk::PhysicalDeviceProperties,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub surface_loader: Surface,
    pub swapchain_loader: Swapchain,
    pub debug_report_loader: Option<DebugReport>,
    pub debug_call_back: Option<vk::DebugReportCallbackEXT>,

    pub pdevice: vk::PhysicalDevice,
    pub present_queue_family_index: u32,
    pub present_queue: vk::Queue,

    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,

    pub swapchain: Option<VkSwapchain>,
    swapchain_create_info: VkSwapchainCreateInfo,

    pub samplers: [vk::Sampler; 1],

    pub allocator: vk_mem::Allocator,
    pub bindless_buffers_descriptor_set: vk::DescriptorSet,
    pub bindless_images_descriptor_set: vk::DescriptorSet,

    // TODO: Those also guard updates to the descriptor sets
    pub bindless_buffers_next_descriptor: Mutex<u32>,
    pub bindless_images_next_descriptor: Mutex<u32>,

    pub frame_data: Vec<VkFrameData>,

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
struct VkSwapchainCreateInfo {
    surface_format: vk::SurfaceFormatKHR,
    surface_resolution: vk::Extent2D,
    vsync: bool,
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
            for img in self.present_image_views.drain(..) {
                vk_device().destroy_image_view(img, None);
            }

            // TODO: semaphores

            vk_all()
                .swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

// TODO: Result
fn create_swapchain(
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

impl VkKitchenSink {
    pub fn new(
        window: &winit::Window,
        graphics_debugging: bool,
        vsync: bool,
    ) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let entry = ash::Entry::new()?;
            let surface_extensions = ash_window::enumerate_required_extensions(window)?;
            let instance_extensions = surface_extensions
                .iter()
                .map(|ext| ext.as_ptr())
                .chain(extension_names(graphics_debugging).into_iter())
                .collect::<Vec<_>>();

            let mut layer_names = Vec::new();
            if graphics_debugging {
                layer_names.push(CString::new("VK_LAYER_LUNARG_standard_validation").unwrap());
            }

            let layers_names_raw: Vec<*const i8> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let app_desc = vk::ApplicationInfo::builder().api_version(vk::make_version(1, 0, 0));

            let instance_desc = vk::InstanceCreateInfo::builder()
                .application_info(&app_desc)
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&instance_extensions);

            let instance = entry.create_instance(&instance_desc, None)?;

            let debug_info = vk::DebugReportCallbackCreateInfoEXT::builder()
                .flags(
                    vk::DebugReportFlagsEXT::ERROR
                        | vk::DebugReportFlagsEXT::WARNING
                        | vk::DebugReportFlagsEXT::PERFORMANCE_WARNING,
                )
                .pfn_callback(Some(vulkan_debug_callback));

            let debug_report_loader;
            let debug_call_back;

            if graphics_debugging {
                let loader = DebugReport::new(&entry, &instance);
                debug_call_back = Some(
                    loader
                        .create_debug_report_callback(&debug_info, None)
                        .unwrap(),
                );
                debug_report_loader = Some(loader);
            } else {
                debug_report_loader = None;
                debug_call_back = None;
            }

            // Create a surface from winit window.
            let surface = ash_window::create_surface(&entry, &instance, window, None)?;

            let pdevices = instance
                .enumerate_physical_devices()
                .expect("Physical device error");
            let surface_loader = Surface::new(&entry, &instance);
            let (pdevice, present_queue_family_index) = pdevices
                .iter()
                .map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .filter_map(|(index, ref info)| {
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *pdevice,
                                            index as u32,
                                            surface,
                                        )
                                        .unwrap();
                            match supports_graphic_and_surface {
                                true => Some((*pdevice, index)),
                                _ => None,
                            }
                        })
                        .nth(0)
                })
                .filter_map(|v| v)
                .nth(0)
                .expect("Couldn't find suitable device.");

            let present_queue_family_index = present_queue_family_index as u32;
            let device_properties = instance.get_physical_device_properties(pdevice);
            let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);

            let device_extension_names_raw = vec![
                Swapchain::name().as_ptr(),
                //RayTracing::name().as_ptr(),
                vk::ExtDescriptorIndexingFn::name().as_ptr(),
                vk::ExtScalarBlockLayoutFn::name().as_ptr(),
                vk::KhrMaintenance1Fn::name().as_ptr(),
                vk::KhrMaintenance2Fn::name().as_ptr(),
                vk::KhrMaintenance3Fn::name().as_ptr(),
                vk::KhrGetMemoryRequirements2Fn::name().as_ptr(),
                vk::ExtDescriptorIndexingFn::name().as_ptr(),
                vk::KhrImagelessFramebufferFn::name().as_ptr(),
                vk::KhrImageFormatListFn::name().as_ptr(),
            ];

            let priorities = [1.0];

            let queue_info = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(present_queue_family_index)
                .queue_priorities(&priorities)
                .build()];

            let mut scalar_block = vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT::builder()
                .scalar_block_layout(true)
                .build();

            let mut descriptor_indexing =
                vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::builder()
                    .descriptor_binding_variable_descriptor_count(true)
                    .descriptor_binding_update_unused_while_pending(true)
                    .descriptor_binding_partially_bound(true)
                    .runtime_descriptor_array(true)
                    .shader_uniform_texel_buffer_array_dynamic_indexing(true)
                    .shader_uniform_texel_buffer_array_non_uniform_indexing(true)
                    .shader_sampled_image_array_non_uniform_indexing(true)
                    .build();

            let mut imageless_framebuffer =
                vk::PhysicalDeviceImagelessFramebufferFeaturesKHR::builder()
                    .imageless_framebuffer(true)
                    .build();

            let mut features2 = vk::PhysicalDeviceFeatures2::default();
            instance
                .fp_v1_1()
                .get_physical_device_features2(pdevice, &mut features2);

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&device_extension_names_raw)
                .enabled_features(&features2.features)
                .push_next(&mut scalar_block)
                .push_next(&mut descriptor_indexing)
                .push_next(&mut imageless_framebuffer)
                .build();

            let device: Device = instance
                .create_device(pdevice, &device_create_info, None)
                .unwrap();

            let present_queue = device.get_device_queue(present_queue_family_index as u32, 0);

            let physical_dimensions = window
                .get_inner_size()
                .unwrap()
                .to_physical(window.get_hidpi_factor());

            let surface_formats = surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap();
            let surface_format = surface_formats
                .iter()
                .map(|sfmt| match sfmt.format {
                    vk::Format::UNDEFINED => vk::SurfaceFormatKHR {
                        format: vk::Format::B8G8R8_UNORM,
                        color_space: sfmt.color_space,
                    },
                    _ => sfmt.clone(),
                })
                .nth(0)
                .expect("Unable to find suitable surface format.");

            let swapchain_loader = Swapchain::new(&instance, &device);

            let swapchain_create_info = VkSwapchainCreateInfo {
                surface_format,
                surface_resolution: vk::Extent2D {
                    width: physical_dimensions.width as u32,
                    height: physical_dimensions.height as u32,
                },
                vsync,
            };
            let swapchain = create_swapchain(
                &device,
                pdevice,
                &swapchain_loader,
                &surface_loader,
                surface,
                swapchain_create_info,
            )
            .unwrap();

            let sampler_info = vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                address_mode_u: vk::SamplerAddressMode::REPEAT,
                address_mode_v: vk::SamplerAddressMode::REPEAT,
                address_mode_w: vk::SamplerAddressMode::REPEAT,
                max_anisotropy: 1.0,
                border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
                compare_op: vk::CompareOp::NEVER,
                ..Default::default()
            };
            let sampler = device.create_sampler(&sampler_info, None).unwrap();

            let allocator_info = vk_mem::AllocatorCreateInfo {
                physical_device: pdevice,
                device: device.clone(),
                instance: instance.clone(),
                flags: vk_mem::AllocatorCreateFlags::NONE,
                preferred_large_heap_block_size: 0,
                frame_in_use_count: swapchain.present_images.len() as u32,
                heap_size_limits: None,
            };
            let allocator = vk_mem::Allocator::new(&allocator_info)
                .expect("failed to create vulkan memory allocator");

            let bindless_buffers_descriptor_set = Self::create_bindless_resource_descriptor_set(
                &device,
                vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            );
            let bindless_images_descriptor_set = Self::create_bindless_resource_descriptor_set(
                &device,
                vk::DescriptorType::SAMPLED_IMAGE,
            );

            let depth_image_create_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D24_UNORM_S8_UINT)
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

            vk_add_setup_command(move |vk_all, vk_frame| {
                let cb = vk_frame.command_buffer.lock().unwrap();
                let cb: vk::CommandBuffer = cb.cb;

                {
                    vk_all.record_image_aspect_barrier(
                        cb,
                        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
                        ImageBarrier::new(
                            depth_image,
                            vk_sync::AccessType::Nothing,
                            vk_sync::AccessType::DepthAttachmentWriteStencilReadOnly,
                        )
                        .with_discard(true),
                    )
                };
            });

            let mut res = Self {
                entry,
                instance,
                device,
                device_properties,
                device_memory_properties,
                present_queue_family_index,
                pdevice,
                surface_loader,
                surface_format,
                present_queue,
                swapchain_loader,
                swapchain: Some(swapchain),
                swapchain_create_info,
                samplers: [sampler],
                frame_data: Vec::new(),
                surface,
                allocator,
                bindless_buffers_descriptor_set,
                bindless_buffers_next_descriptor: Mutex::new(0),
                bindless_images_descriptor_set,
                bindless_images_next_descriptor: Mutex::new(0),
                debug_call_back,
                debug_report_loader,
                depth_image,
                depth_image_view,
            };

            res.create_frame_data();
            Ok(res)
        }
    }

    pub fn swapchain_size_pixels(&self) -> (u32, u32) {
        let vk::Extent2D { width, height } = self.swapchain.as_ref().unwrap().surface_resolution;
        (width, height)
    }

    fn create_frame_data(&mut self) {
        self.frame_data = (0..self.swapchain.as_ref().unwrap().present_images.len())
            .map(|_| {
                let uniforms = LinearUniformBuffer::new(
                    1 << 20,
                    &self.allocator,
                    self.device_properties
                        .limits
                        .min_uniform_buffer_offset_alignment as usize,
                );

                let submit_done_fence = unsafe {
                    self.device.create_fence(
                        &vk::FenceCreateInfo::builder()
                            .flags(vk::FenceCreateFlags::SIGNALED)
                            .build(),
                        None,
                    )
                }
                .expect("Create fence failed.");

                let profiler_data = VkProfilerData::new(&self.device, &self.allocator);

                VkFrameData {
                    uniforms,
                    descriptor_pool: Mutex::new(allocate_frame_descriptor_pool(&self.device)),
                    command_buffer: Mutex::new(allocate_frame_command_buffer(
                        &self.device,
                        self.present_queue_family_index,
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

        unsafe {
            self.device.update_descriptor_sets(
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

        unsafe {
            self.device.update_descriptor_sets(
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

            let descriptor_pool = self
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap();

            let descriptor_sets = self
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

    fn create_bindless_resource_descriptor_set(
        device: &Device,
        descriptor_type: vk::DescriptorType,
    ) -> vk::DescriptorSet {
        let desc_count = 1 << 18;

        unsafe {
            let binding_flags = [vk::DescriptorBindingFlagsEXT::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlagsEXT::PARTIALLY_BOUND
                | vk::DescriptorBindingFlagsEXT::UPDATE_UNUSED_WHILE_PENDING];
            let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
                .binding_flags(&binding_flags)
                .build();

            let descriptor_set_layout = {
                device
                    .create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::builder()
                            .bindings(&[vk::DescriptorSetLayoutBinding::builder()
                                .descriptor_count(desc_count)
                                .descriptor_type(descriptor_type)
                                .stage_flags(
                                    vk::ShaderStageFlags::COMPUTE
                                        | vk::ShaderStageFlags::VERTEX
                                        | vk::ShaderStageFlags::FRAGMENT,
                                )
                                .binding(0)
                                .build()])
                            .push_next(&mut binding_flags)
                            .build(),
                        None,
                    )
                    .unwrap()
            };

            let descriptor_sizes = [vk::DescriptorPoolSize {
                ty: descriptor_type,
                descriptor_count: desc_count,
            }];

            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&descriptor_sizes)
                .max_sets(1);

            let descriptor_pool = device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap();

            let descriptor_sets = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&vec![descriptor_set_layout])
                        .build(),
                )
                .unwrap();

            descriptor_sets[0]
        }
    }

    pub fn record_image_barrier(&self, cb: vk::CommandBuffer, barrier: ImageBarrier) {
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        vk_sync::cmd::pipeline_barrier(
            self.device.fp_v1_0(),
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
        &self,
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
            self.device.fp_v1_0(),
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
}

impl Drop for VkKitchenSink {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
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
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            if let Some(debug_report_loader) = self.debug_report_loader.as_ref() {
                debug_report_loader
                    .destroy_debug_report_callback(self.debug_call_back.unwrap(), None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

pub fn record_submit_commandbuffer<F: FnOnce(&Device)>(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    submit_queue: vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    render_fn: F,
) {
    unsafe {
        device
            .wait_for_fences(&[vk_frame().submit_done_fence], true, std::u64::MAX)
            .expect("Wait for fence failed.");

        let (query_ids, timing_pairs) = vk_frame().profiler_data.retrieve_previous_result();

        crate::gpu_profiler::report_durations_ticks(timing_pairs.chunks_exact(2).enumerate().map(
            |(pair_idx, chunk)| -> (GpuProfilerQueryId, u64) {
                (query_ids[pair_idx], chunk[1] - chunk[0])
            },
        ));

        device
            .reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
            .expect("Reset command buffer failed.");

        {
            let vk_all = vk_all();
            for f in vk_frame().frame_cleanup.lock().unwrap().drain(..) {
                (f)(vk_all);
            }
        }

        {
            let pool = vk_frame().descriptor_pool.lock().unwrap();
            device
                .reset_descriptor_pool(*pool, Default::default())
                .expect("reset_descriptor_pool");
            drop(pool);
        }

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Begin commandbuffer");

        vk_frame().profiler_data.begin_frame(device, command_buffer);
        vk_frame_mut().uniforms.map();

        for f in VK_SETUP_COMMANDS.lock().unwrap().drain(..).into_iter() {
            f(vk_all(), vk_frame());
        }

        render_fn(device);

        for fd in VK_KITCHEN_SINK.as_mut().unwrap().frame_data.iter_mut() {
            fd.uniforms.unmap();
        }

        vk_frame()
            .profiler_data
            .finish_frame(device, command_buffer);

        device
            .end_command_buffer(command_buffer)
            .expect("End commandbuffer");

        let submit_fence = vk_frame().submit_done_fence;
        device.reset_fences(&[submit_fence]).expect("reset_fences");

        let command_buffers = vec![command_buffer];

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);

        device
            .queue_submit(submit_queue, &[submit_info.build()], submit_fence)
            .expect("queue submit failed.");
    }
}

static mut VK_KITCHEN_SINK: Option<VkKitchenSink> = None;
static mut VK_CURRENT_FRAME_DATA_IDX: usize = std::usize::MAX;

pub fn initialize_vk_state(vk: VkKitchenSink) {
    unsafe {
        VK_KITCHEN_SINK = Some(vk);
    }
}

pub fn vk_device() -> &'static Device {
    unsafe { std::mem::transmute(&VK_KITCHEN_SINK.as_ref().expect("vk kitchen sink").device) }
}

pub unsafe fn vk_begin_frame(data_idx: usize) {
    VK_CURRENT_FRAME_DATA_IDX = data_idx;
}

pub unsafe fn vk_all() -> &'static VkKitchenSink {
    std::mem::transmute(VK_KITCHEN_SINK.as_ref().expect("vk kitchen sink"))
}

pub unsafe fn vk_resize(width: u32, height: u32) -> bool {
    let vk = VK_KITCHEN_SINK.as_mut().unwrap();
    vk.device.device_wait_idle().unwrap();

    let mut create_info = vk.swapchain_create_info;
    create_info.surface_resolution = vk::Extent2D { width, height };

    vk.frame_data = Vec::new();
    vk.swapchain = None;

    vk.swapchain = create_swapchain(
        &vk.device,
        vk.pdevice,
        &vk.swapchain_loader,
        &vk.surface_loader,
        vk.surface,
        create_info,
    );

    if vk.swapchain.is_some() {
        vk.swapchain_create_info = create_info;
        vk.create_frame_data();
        true
    } else {
        false
    }
}

pub unsafe fn vk_frame() -> &'static VkFrameData {
    assert!(VK_CURRENT_FRAME_DATA_IDX != std::usize::MAX);
    std::mem::transmute(
        &VK_KITCHEN_SINK
            .as_ref()
            .expect("vk kitchen sink")
            .frame_data[VK_CURRENT_FRAME_DATA_IDX],
    )
}

pub fn vk_frame_index() -> usize {
    unsafe { VK_CURRENT_FRAME_DATA_IDX }
}

pub(crate) unsafe fn vk_frame_mut() -> &'static mut VkFrameData {
    assert!(VK_CURRENT_FRAME_DATA_IDX != std::usize::MAX);
    std::mem::transmute(
        &mut VK_KITCHEN_SINK
            .as_mut()
            .expect("vk kitchen sink")
            .frame_data[VK_CURRENT_FRAME_DATA_IDX],
    )
}

lazy_static! {
    pub(crate) static ref VK_SETUP_COMMANDS: Mutex<Vec<Box<dyn FnOnce(&VkKitchenSink, &'static VkFrameData) + Send + 'static>>> =
        { Mutex::new(Vec::new()) };
}

pub fn vk_add_setup_command(f: impl FnOnce(&VkKitchenSink, &'static VkFrameData) + Send + 'static) {
    if unsafe { VK_CURRENT_FRAME_DATA_IDX } == std::usize::MAX {
        // If we haven't started rendering yet, delay this.
        VK_SETUP_COMMANDS.lock().unwrap().push(Box::new(f));
    } else {
        // Otherwise do it now
        unsafe { f(vk_all(), vk_frame()) };
    }
}
