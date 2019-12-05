//use ash::extensions::nv::RayTracing;
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
    println!("{:?}", CStr::from_ptr(p_message));
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
    //pool: vk::CommandPool,
}

pub struct LinearUniformBuffer {
    write_head: std::sync::atomic::AtomicUsize,
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    //allocation_info: vk_mem::AllocationInfo,
    size: vk::DeviceSize,
    mapped_ptr: *mut u8,
    min_offset_alignment: usize,
}

unsafe impl Send for LinearUniformBuffer {}
unsafe impl Sync for LinearUniformBuffer {}

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

        let (buffer, allocation, _allocation_info) =
            allocator.create_buffer(&buffer_info, &mem_info).unwrap();

        let mapped_ptr: *mut u8 = allocator.map_memory(&allocation).expect("map_memory");

        Self {
            write_head: std::sync::atomic::AtomicUsize::new(0),
            buffer,
            allocation,
            //allocation_info,
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
            unsafe {
                vk_all()
                    .allocator
                    .unmap_memory(&self.allocation)
                    .expect("unmap_memory");
            }
            self.mapped_ptr = std::ptr::null_mut();
            self.write_head
                .store(0, std::sync::atomic::Ordering::Relaxed);
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

pub struct VkFrameData {
    pub uniforms: LinearUniformBuffer,
    pub descriptor_pool: Mutex<vk::DescriptorPool>,
    pub command_buffer: Mutex<VkCommandBufferData>,
    pub present_image: vk::Image,
    pub present_image_view: vk::ImageView,
    pub rendering_complete_semaphore: vk::Semaphore,
    pub submit_done_fence: vk::Fence,
    pub frame_cleanup: Mutex<Vec<Box<dyn Fn(&VkKitchenSink) + Send + Sync>>>,
}

pub const SAMPLER_LINEAR: usize = 0;

pub struct VkKitchenSink {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub device_properties: vk::PhysicalDeviceProperties,
    pub surface_loader: Surface,
    pub swapchain_loader: Swapchain,
    pub debug_report_loader: Option<DebugReport>,
    pub debug_call_back: Option<vk::DebugReportCallbackEXT>,

    pub pdevice: vk::PhysicalDevice,
    pub present_queue_family_index: u32,
    pub present_queue: vk::Queue,

    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,

    pub swapchain: vk::SwapchainKHR,
    pub swapchain_acquired_semaphores: Vec<vk::Semaphore>,

    pub samplers: [vk::Sampler; 1],

    pub allocator: vk_mem::Allocator,
    pub bindless_buffers_descriptor_set: vk::DescriptorSet,
    pub bindless_buffers_next_descriptor: std::sync::atomic::AtomicU32,
    pub bindless_images_descriptor_set: vk::DescriptorSet,
    pub bindless_images_next_descriptor: std::sync::atomic::AtomicU32,

    pub frame_data: Vec<VkFrameData>,

    pub window_width: u32,
    pub window_height: u32,

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

    VkCommandBufferData {
        cb,
        //pool,
    }
}

impl VkKitchenSink {
    pub fn new(window: &winit::Window, graphics_debugging: bool) -> Result<Self, Box<dyn Error>> {
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
            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(pdevice, surface)
                .unwrap();
            let mut desired_image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }
            let physical_dimensions = window
                .get_inner_size()
                .unwrap()
                .to_physical(window.get_hidpi_factor());
            let surface_resolution = match surface_capabilities.current_extent.width {
                std::u32::MAX => vk::Extent2D {
                    width: physical_dimensions.width as u32,
                    height: physical_dimensions.height as u32,
                },
                _ => surface_capabilities.current_extent,
            };
            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(pdevice, surface)
                .unwrap();
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);
            let swapchain_loader = Swapchain::new(&instance, &device);

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_resolution.clone())
                .image_usage(vk::ImageUsageFlags::STORAGE)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1);

            let swapchain = swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();

            let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
            let present_image_views: Vec<vk::ImageView> = present_images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
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
                    device.create_image_view(&create_view_info, None).unwrap()
                })
                .collect();

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let create_swapchain_semaphores = || -> Vec<vk::Semaphore> {
                (0..present_images.len())
                    .map(|_| {
                        device
                            .create_semaphore(&semaphore_create_info, None)
                            .unwrap()
                    })
                    .collect()
            };

            let sampler_info = vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                max_anisotropy: 1.0,
                border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
                compare_op: vk::CompareOp::NEVER,
                ..Default::default()
            };
            let sampler = device.create_sampler(&sampler_info, None).unwrap();

            let swapchain_acquired_semaphores = create_swapchain_semaphores();
            let rendering_complete_semaphores = create_swapchain_semaphores();

            let allocator_info = vk_mem::AllocatorCreateInfo {
                physical_device: pdevice,
                device: device.clone(),
                instance: instance.clone(),
                flags: vk_mem::AllocatorCreateFlags::NONE,
                preferred_large_heap_block_size: 0,
                frame_in_use_count: surface_capabilities.min_image_count,
                heap_size_limits: None,
            };
            let allocator = vk_mem::Allocator::new(&allocator_info)
                .expect("failed to create vulkan memory allocator");

            let frame_data = (0..present_images.len())
                .map(|i| {
                    let uniforms = LinearUniformBuffer::new(
                        1 << 20,
                        &allocator,
                        device_properties.limits.min_uniform_buffer_offset_alignment as usize,
                    );

                    let submit_done_fence = device
                        .create_fence(
                            &vk::FenceCreateInfo::builder()
                                .flags(vk::FenceCreateFlags::SIGNALED)
                                .build(),
                            None,
                        )
                        .expect("Create fence failed.");

                    VkFrameData {
                        uniforms,
                        descriptor_pool: Mutex::new(allocate_frame_descriptor_pool(&device)),
                        command_buffer: Mutex::new(allocate_frame_command_buffer(
                            &device,
                            present_queue_family_index,
                        )),
                        present_image: present_images[i],
                        present_image_view: present_image_views[i],
                        rendering_complete_semaphore: rendering_complete_semaphores[i],
                        submit_done_fence,
                        frame_cleanup: Mutex::new(Default::default()),
                    }
                })
                .collect();

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
                    width: surface_resolution.width,
                    height: surface_resolution.height,
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

            Ok(Self {
                entry,
                instance,
                device,
                device_properties,
                present_queue_family_index,
                pdevice,
                surface_loader,
                surface_format,
                present_queue,
                surface_resolution,
                swapchain_loader,
                swapchain,
                swapchain_acquired_semaphores,
                samplers: [sampler],
                frame_data,
                surface,
                allocator,
                bindless_buffers_descriptor_set,
                bindless_buffers_next_descriptor: std::sync::atomic::AtomicU32::new(0),
                bindless_images_descriptor_set,
                bindless_images_next_descriptor: std::sync::atomic::AtomicU32::new(0),
                debug_call_back,
                debug_report_loader,
                window_width: physical_dimensions.width as u32,
                window_height: physical_dimensions.height as u32,
                depth_image,
                depth_image_view,
            })
        }
    }

    pub(crate) fn register_image_bindless_index(&self, view: vk::ImageView) -> u32 {
        let idx = self
            .bindless_images_next_descriptor
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

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
        let idx = self
            .bindless_buffers_next_descriptor
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

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
                    descriptor_count: self.frame_data.len() as u32,
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
            for s in self.swapchain_acquired_semaphores.iter() {
                self.device.destroy_semaphore(*s, None);
            }
            for s in self.frame_data.iter() {
                self.device
                    .destroy_semaphore(s.rendering_complete_semaphore, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
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

        vk_frame_mut().uniforms.map();

        for f in VK_SETUP_COMMANDS.lock().unwrap().drain(..).into_iter() {
            f(vk_all(), vk_frame());
        }

        render_fn(device);

        for fd in VK_KITCHEN_SINK.as_mut().unwrap().frame_data.iter_mut() {
            fd.uniforms.unmap();
        }

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

pub unsafe fn vk_frame() -> &'static VkFrameData {
    assert!(VK_CURRENT_FRAME_DATA_IDX != std::usize::MAX);
    std::mem::transmute(
        &VK_KITCHEN_SINK
            .as_ref()
            .expect("vk kitchen sink")
            .frame_data[VK_CURRENT_FRAME_DATA_IDX],
    )
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
