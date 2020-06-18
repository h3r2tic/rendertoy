use super::transient_resource::*;
use crate::{vk, vulkan::*};
use ash::version::DeviceV1_0;

#[derive(Eq, PartialEq, Hash, Clone, Copy, Serialize, Debug)]
pub enum TextureType {
    Type2D,
    Type3D,
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Serialize, Debug)]
pub struct TextureKey {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: i32,
    pub tex_type: TextureType,
}

impl TextureKey {
    pub fn new(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
            width,
            height,
            depth: 1,
            format: format.as_raw(),
            tex_type: TextureType::Type2D,
        }
    }

    pub fn new_3d(width: u32, height: u32, depth: u32, format: vk::Format) -> Self {
        Self {
            width,
            height,
            depth,
            format: format.as_raw(),
            tex_type: TextureType::Type3D,
        }
    }

    pub fn res_div_round_up(&self, x: u32, y: u32) -> Self {
        let mut res = self.clone();
        res.width = (res.width + x - 1) / x;
        res.height = (res.height + y - 1) / y;
        res
    }

    pub fn res_div_round_up_3d(&self, x: u32, y: u32, z: u32) -> Self {
        let mut res = self.clone();
        res.width = (res.width + x - 1) / x;
        res.height = (res.height + y - 1) / y;
        res.depth = (res.depth + z - 1) / z;
        res
    }

    pub fn padded(&self, x: u32, y: u32) -> Self {
        let mut res = self.clone();
        res.width += x;
        res.height += y;
        res
    }

    pub fn half_res(&self) -> Self {
        self.res_div_round_up_3d(2, 2, 2)
    }

    pub fn with_width(&self, v: u32) -> Self {
        let mut res = self.clone();
        res.width = v;
        res
    }

    pub fn with_height(&self, v: u32) -> Self {
        let mut res = self.clone();
        res.height = v;
        res
    }

    pub fn with_depth(&self, v: u32) -> Self {
        let mut res = self.clone();
        res.depth = v;
        res
    }

    pub fn with_format(&self, format: vk::Format) -> Self {
        let mut res = self.clone();
        res.format = format.as_raw();
        res
    }
}

#[derive(Clone)]
pub struct Texture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub rt_view: vk::ImageView,
    pub storage_view: vk::ImageView,
    pub key: TextureKey,
    pub bindless_index: u32,
    _allocation: SharedTransientAllocation,
}

#[derive(Clone)]
pub struct ImageResource {
    image: vk::Image,
    view: vk::ImageView,
    rt_view: vk::ImageView,
    storage_view: vk::ImageView,
    memory: vk::DeviceMemory,
    bindless_index: u32,

    // Validation layers retain pointers to those, so we must keep them valid :/
    format_list: Option<Box<vk::ImageFormatListCreateInfoKHR>>,
    view_formats: Option<Vec<vk::Format>>,
}

unsafe impl Send for ImageResource {}
unsafe impl Sync for ImageResource {}

impl ImageResource {
    fn new() -> Self {
        ImageResource {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            rt_view: vk::ImageView::null(),
            storage_view: vk::ImageView::null(),
            memory: vk::DeviceMemory::null(),
            bindless_index: std::u32::MAX,
            format_list: None,
            view_formats: None,
        }
    }

    fn create_image(
        &mut self,
        image_type: vk::ImageType,
        format: vk::Format,
        storage_format: vk::Format,
        extent: vk::Extent3D,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
    ) {
        let mem_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let mut view_formats = vec![format];
        if storage_format != format {
            view_formats.push(storage_format);
        }

        let mut format_list = Box::new(
            vk::ImageFormatListCreateInfoKHR::builder()
                .view_formats(&view_formats)
                .build(),
        );

        let create_info = vk::ImageCreateInfo::builder()
            .image_type(image_type)
            .format(storage_format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(tiling)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
            .push_next(&mut *format_list)
            .build();

        let (image, _allocation, _allocation_info) = vk()
            .allocator
            .create_image(&create_info, &mem_info)
            .unwrap();

        self.view_formats = Some(view_formats);
        self.format_list = Some(format_list);
        self.image = image;
    }

    fn create_view(
        &mut self,
        view_type: vk::ImageViewType,
        format: vk::Format,
        storage_format: vk::Format,
        readonly_usage: vk::ImageUsageFlags,
        rt_usage: vk::ImageUsageFlags,
        range: vk::ImageSubresourceRange,
    ) {
        let image = self.image;
        let create_info = || {
            vk::ImageViewCreateInfo::builder()
                .view_type(view_type)
                .format(format)
                .subresource_range(range)
                .image(image)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
        };

        let device = vk().device.clone();

        {
            let mut view_usage = vk::ImageViewUsageCreateInfo::builder().usage(readonly_usage);
            let create_info = create_info().push_next(&mut view_usage).build();
            self.view = unsafe { device.create_image_view(&create_info, None).unwrap() };
        }

        {
            let mut view_usage = vk::ImageViewUsageCreateInfo::builder().usage(rt_usage);
            let create_info = create_info().push_next(&mut view_usage).build();
            self.rt_view = unsafe { device.create_image_view(&create_info, None).unwrap() };
        }

        {
            let create_info = create_info()
                .format(storage_format)
                .image(self.image)
                .build();
            self.storage_view = unsafe { device.create_image_view(&create_info, None).unwrap() };
        }
    }
}

pub fn create_texture(key: TextureKey) -> Texture {
    create_transient(key)
}

fn get_storage_compatible_format(f: vk::Format) -> vk::Format {
    match f {
        vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
        _ => f,
    }
}

impl TransientResource for Texture {
    type Desc = TextureKey;
    type Allocation = ImageResource;

    fn new(
        desc: TextureKey,
        allocation: std::sync::Arc<TransientResourceAllocation<TextureKey, Self::Allocation>>,
    ) -> Self {
        Self {
            image: allocation.payload.image,
            view: allocation.payload.view,
            rt_view: allocation.payload.rt_view,
            storage_view: allocation.payload.storage_view,
            key: desc,
            bindless_index: allocation.payload.bindless_index,
            _allocation: allocation,
        }
    }

    fn allocate_payload(key: TextureKey) -> Self::Allocation {
        let format = vk::Format::from_raw(key.format);
        let mut img = ImageResource::new();
        let storage_format = get_storage_compatible_format(format);
        img.create_image(
            match key.tex_type {
                TextureType::Type2D => vk::ImageType::TYPE_2D,
                TextureType::Type3D => vk::ImageType::TYPE_3D,
            },
            format,
            storage_format,
            vk::Extent3D::builder()
                .width(key.width)
                .height(key.height)
                .depth(key.depth)
                .build(),
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        );

        img.create_view(
            match key.tex_type {
                TextureType::Type2D => vk::ImageViewType::TYPE_2D,
                TextureType::Type3D => vk::ImageViewType::TYPE_3D,
            },
            format,
            storage_format,
            vk::ImageUsageFlags::SAMPLED,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        );

        img.bindless_index = vk_state().register_image_bindless_index(img.view);

        img
    }
}
