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

pub struct VkRenderDevice {
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

    pub allocator: vk_mem::Allocator,
    pub samplers: [vk::Sampler; 1], // immutable
}

impl VkRenderDevice {
    pub(crate) fn new(
        window: &winit::Window,
        graphics_debugging: bool,
        device_index: usize,
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
                .nth(device_index)
                .expect("Couldn't find suitable device.");

            let present_queue_family_index = present_queue_family_index as u32;
            let device_properties = instance.get_physical_device_properties(pdevice);
            unsafe {
                tracing::info!(
                    "Using device {:?}",
                    CStr::from_ptr(device_properties.device_name.as_ptr())
                );
            }

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
                //vk::KhrImagelessFramebufferFn::name().as_ptr(),
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

            let allocator_info = vk_mem::AllocatorCreateInfo {
                physical_device: pdevice,
                device: device.clone(),
                instance: instance.clone(),
                flags: vk_mem::AllocatorCreateFlags::NONE,
                preferred_large_heap_block_size: 0,
                frame_in_use_count: 0,
                heap_size_limits: None,
            };
            let allocator = vk_mem::Allocator::new(&allocator_info)
                .expect("failed to create vulkan memory allocator");

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

            let swapchain_loader = Swapchain::new(&instance, &device);

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

            Ok(Self {
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
                allocator,
                samplers: [sampler],
                debug_call_back,
                debug_report_loader,
                surface,
            })
        }
    }

    pub(crate) fn create_bindless_resource_descriptor_set(
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
}
