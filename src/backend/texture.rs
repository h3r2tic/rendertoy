use super::transient_resource::*;
use crate::{vk, vulkan::*};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0, InstanceV1_1};

#[derive(Eq, PartialEq, Hash, Clone, Copy, Serialize, Debug)]
pub struct TextureKey {
    pub width: u32,
    pub height: u32,
    pub format: i32,
}

impl TextureKey {
    pub fn new(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
            width,
            height,
            format: format.as_raw(),
        }
    }

    pub fn res_div_round_up(&self, x: u32, y: u32) -> Self {
        let mut res = self.clone();
        res.width = (res.width + x - 1) / x;
        res.height = (res.height + y - 1) / y;
        res
    }

    pub fn padded(&self, x: u32, y: u32) -> Self {
        let mut res = self.clone();
        res.width += x;
        res.height += y;
        res
    }

    pub fn half_res(&self) -> Self {
        self.res_div_round_up(2, 2)
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
    pub storage_view: vk::ImageView,
    pub key: TextureKey,
    _allocation: SharedTransientAllocation,
}

#[derive(Clone)]
pub struct ImageResource {
    image: vk::Image,
    view: vk::ImageView,
    storage_view: vk::ImageView,
    memory: vk::DeviceMemory,
}

impl ImageResource {
    fn new() -> Self {
        ImageResource {
            image: vk::Image::null(),
            view: vk::ImageView::null(),

            storage_view: vk::ImageView::null(),

            memory: vk::DeviceMemory::null(),
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
        memory_flags: vk::MemoryPropertyFlags,
    ) {
        unsafe {
            let mut create_info = vk::ImageCreateInfo::builder()
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
                .build();

            self.image = vk_device().create_image(&create_info, None).unwrap();

            let requirements = vk_device().get_image_memory_requirements(self.image);
            let memory_index = find_memorytype_index(
                &requirements,
                &vk_all().device_memory_properties,
                memory_flags,
            )
            .expect("Unable to find suitable memory index image.");

            let allocate_info = vk::MemoryAllocateInfo {
                allocation_size: requirements.size,
                memory_type_index: memory_index,
                ..Default::default()
            };

            self.memory = vk_device().allocate_memory(&allocate_info, None).unwrap();

            vk_device()
                .bind_image_memory(self.image, self.memory, 0)
                .expect("Unable to bind image memory");
        }
    }

    fn create_view(
        &mut self,
        view_type: vk::ImageViewType,
        format: vk::Format,
        storage_format: vk::Format,
        storage_usage: vk::ImageUsageFlags,
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

        {
            let mut view_usage = vk::ImageViewUsageCreateInfo::builder().usage(storage_usage);

            let create_info = create_info().push_next(&mut view_usage).build();
            self.view = unsafe { vk_device().create_image_view(&create_info, None).unwrap() };
        }

        {
            let create_info = create_info()
                .format(storage_format)
                .image(self.image)
                .build();
            self.storage_view =
                unsafe { vk_device().create_image_view(&create_info, None).unwrap() };
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
            storage_view: allocation.payload.storage_view,
            // TODO: sampler
            key: desc,
            _allocation: allocation,
        }
    }

    fn allocate_payload(key: TextureKey) -> Self::Allocation {
        let format = vk::Format::from_raw(key.format);
        let mut img = ImageResource::new();
        let storage_format = get_storage_compatible_format(format);
        img.create_image(
            vk::ImageType::TYPE_2D,
            format,
            storage_format,
            vk::Extent3D::builder()
                .width(key.width)
                .height(key.height)
                .depth(1)
                .build(),
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        img.create_view(
            vk::ImageViewType::TYPE_2D,
            format,
            storage_format,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        );

        img
        /*unsafe {
            let mut prev_bound_texture = 0;
            gl.GetIntegerv(gl::TEXTURE_BINDING_2D, &mut prev_bound_texture);

            let mut texture_id = 0;
            gl.GenTextures(1, &mut texture_id);
            gl.BindTexture(gl::TEXTURE_2D, texture_id);
            gl.TexStorage2D(
                gl::TEXTURE_2D,
                1,
                key.format,
                key.width as i32,
                key.height as i32,
            );
            gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
            gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
            gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);

            let mut sampler_id = 0;
            gl.GenSamplers(1, &mut sampler_id);
            gl.SamplerParameteri(sampler_id, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl.SamplerParameteri(sampler_id, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
            gl.SamplerParameteri(sampler_id, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
            gl.SamplerParameteri(sampler_id, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);

            let bindless_handle = gl.GetTextureHandleARB(texture_id);
            gl.MakeTextureHandleResidentARB(bindless_handle);

            // Restore the previously bound texture
            gl.BindTexture(gl::TEXTURE_2D, prev_bound_texture as u32);

            TextureAllocation {
                texture_id,
                sampler_id,
                bindless_handle,
            }
        }*/
    }
}
