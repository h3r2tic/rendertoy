use ash::extensions::nv::RayTracing;
use ash::extensions::{
    ext::DebugReport,
    khr::{Surface, Swapchain},
};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0, InstanceV1_1};
use ash::{vk, Device, Entry, Instance};
use std::error::Error;
use std::ffi::{CStr, CString};
use std::io::Cursor;
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

fn extension_names() -> Vec<*const i8> {
    vec![
        DebugReport::name().as_ptr(),
        vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
    ]
}

pub struct VkFrameData {
    pub descriptor_pool: Mutex<vk::DescriptorPool>,
    pub present_image: vk::Image,
    pub present_image_view: vk::ImageView,
    pub rendering_complete_semaphore: vk::Semaphore,
}

pub struct VkKitchenSink {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub surface_loader: Surface,
    pub swapchain_loader: Swapchain,
    pub debug_report_loader: DebugReport,
    pub debug_call_back: vk::DebugReportCallbackEXT,

    pub pdevice: vk::PhysicalDevice,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub present_queue_family_index: u32,
    pub present_queue: vk::Queue,

    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,

    pub swapchain: vk::SwapchainKHR,
    pub swapchain_acquired_semaphores: Vec<vk::Semaphore>,

    pub frame_data: Vec<VkFrameData>,

    pub window_width: u32,
    pub window_height: u32,
}

struct ImageBarrier {
    image: vk::Image,
    prev_access: vk_sync::AccessType,
    next_access: vk_sync::AccessType,
    discard: bool,
}

impl ImageBarrier {
    fn new(
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

    fn with_discard(mut self, discard: bool) -> Self {
        self.discard = discard;
        self
    }
}

fn allocate_frame_descriptor_pool(device: &Device) -> vk::DescriptorPool {
    let descriptor_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1 << 20,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1 << 20,
        },
    ];

    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&descriptor_sizes)
        .max_sets(1 << 20);

    unsafe {
        device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap()
    }
}

impl VkKitchenSink {
    pub fn new(window: &winit::Window) -> Result<Self, Box<dyn Error>> {
        unsafe {
            let entry = ash::Entry::new()?;
            let surface_extensions = ash_window::enumerate_required_extensions(window)?;
            let instance_extensions = surface_extensions
                .iter()
                .map(|ext| ext.as_ptr())
                .chain(extension_names().into_iter())
                .collect::<Vec<_>>();
            let layer_names = [CString::new("VK_LAYER_LUNARG_standard_validation").unwrap()];
            let layers_names_raw: Vec<*const i8> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let app_desc =
                vk::ApplicationInfo::builder().api_version(ash::vk_make_version!(1, 0, 0));

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

            let debug_report_loader = DebugReport::new(&entry, &instance);
            let debug_call_back = debug_report_loader
                .create_debug_report_callback(&debug_info, None)
                .unwrap();

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
                                    && surface_loader.get_physical_device_surface_support(
                                        *pdevice,
                                        index as u32,
                                        surface,
                                    );
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

            let device_extension_names_raw = vec![
                Swapchain::name().as_ptr(),
                //RayTracing::name().as_ptr(),
                vk::ExtDescriptorIndexingFn::name().as_ptr(),
                vk::ExtScalarBlockLayoutFn::name().as_ptr(),
                vk::KhrMaintenance3Fn::name().as_ptr(),
                vk::KhrGetMemoryRequirements2Fn::name().as_ptr(),
            ];

            let priorities = [1.0];

            let queue_info = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(present_queue_family_index)
                .queue_priorities(&priorities)
                .build()];

            let mut descriptor_indexing =
                vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::builder()
                    .descriptor_binding_variable_descriptor_count(true)
                    .runtime_descriptor_array(true)
                    .build();

            let mut scalar_block = vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT::builder()
                .scalar_block_layout(true)
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
            let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);

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

            let swapchain_acquired_semaphores = create_swapchain_semaphores();
            let rendering_complete_semaphores = create_swapchain_semaphores();

            let frame_data = (0..present_images.len())
                .map(|i| VkFrameData {
                    descriptor_pool: Mutex::new(allocate_frame_descriptor_pool(&device)),
                    present_image: present_images[i],
                    present_image_view: present_image_views[i],
                    rendering_complete_semaphore: rendering_complete_semaphores[i],
                })
                .collect();

            Ok(Self {
                entry,
                instance,
                device,
                present_queue_family_index,
                pdevice,
                device_memory_properties,
                surface_loader,
                surface_format,
                present_queue,
                surface_resolution,
                swapchain_loader,
                swapchain,
                swapchain_acquired_semaphores,
                frame_data,
                surface,
                debug_call_back,
                debug_report_loader,
                window_width: physical_dimensions.width as u32,
                window_height: physical_dimensions.height as u32,
            })
        }
    }

    fn create_present_descriptor_sets(
        &self,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Vec<vk::DescriptorSet> {
        unsafe {
            let descriptor_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: self.frame_data.len() as u32,
            }];

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

            for (img_idx, ds) in descriptor_sets.iter().enumerate() {
                let image_info = [vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::GENERAL)
                    .image_view(self.frame_data[img_idx].present_image_view)
                    .build()];

                let image_write = vk::WriteDescriptorSet::builder()
                    .dst_set(*ds)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&image_info)
                    .build();

                self.device.update_descriptor_sets(&[image_write], &[]);
            }

            descriptor_sets
        }
    }

    fn create_command_pool(&self) -> vk::CommandPool {
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.present_queue_family_index);

        unsafe {
            self.device
                .create_command_pool(&pool_create_info, None)
                .unwrap()
        }
    }

    fn create_command_buffers(
        &self,
        pool: vk::CommandPool,
        count: usize,
    ) -> Vec<vk::CommandBuffer> {
        unsafe {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(count as u32)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            self.device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap()
        }
    }

    fn record_image_barrier(&self, cb: vk::CommandBuffer, barrier: ImageBarrier) {
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
            self.debug_report_loader
                .destroy_debug_report_callback(self.debug_call_back, None);
            self.instance.destroy_instance(None);
        }
    }
}

pub fn record_submit_commandbuffer<D: DeviceV1_0, F: FnOnce(&D, vk::CommandBuffer)>(
    device: &D,
    command_buffer: vk::CommandBuffer,
    submit_queue: vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    f: F,
) {
    unsafe {
        device
            .reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
            .expect("Reset command buffer failed.");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Begin commandbuffer");
        f(device, command_buffer);
        device
            .end_command_buffer(command_buffer)
            .expect("End commandbuffer");

        let submit_fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .expect("Create fence failed.");

        let command_buffers = vec![command_buffer];

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);

        device
            .queue_submit(submit_queue, &[submit_info.build()], submit_fence)
            .expect("queue submit failed.");
        device
            .wait_for_fences(&[submit_fence], true, std::u64::MAX)
            .expect("Wait for fence failed.");
        device.destroy_fence(submit_fence, None);
    }
}

static mut VK_KITCHEN_SINK: Option<VkKitchenSink> = None;

pub fn initialize_vk_state(vk: VkKitchenSink) {
    unsafe {
        VK_KITCHEN_SINK = Some(vk);
    }
}

pub fn vk_device() -> &'static Device {
    unsafe { std::mem::transmute(&VK_KITCHEN_SINK.as_ref().unwrap().device) }
}
