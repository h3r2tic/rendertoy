pub use crate::backend::texture::{Texture, TextureKey};

use crate::backend::{self};
use crate::blob::{load_blob, AssetPath, Blob};
pub use ash::{vk, vk::Format};

use snoozy::*;

#[derive(Serialize, Debug, PartialEq, Eq, Abomonation, Clone, Copy)]
pub enum TexGamma {
    Linear,
    Srgb,
}

#[derive(Serialize, Debug, Clone, Copy, Abomonation)]
pub struct TexParams {
    pub gamma: TexGamma,
}

#[snoozy]
pub async fn load_tex(ctx: Context, path: &AssetPath) -> Result<Texture> {
    let tex = ctx
        .get(load_tex_with_params(
            path.clone(),
            TexParams {
                gamma: TexGamma::Srgb,
            },
        ))
        .await?;
    Ok((*tex).clone())
}

#[derive(Abomonation, Clone)]
pub struct RawRgba8Image {
    data: Vec<u8>,
    dimensions: (u32, u32),
}

#[snoozy(cache)]
pub async fn load_raw_ldr_tex(ctx: Context, path: &AssetPath) -> Result<RawRgba8Image> {
    use image::GenericImageView;

    let blob = ctx.get(&load_blob(path.clone())).await?;

    let image = image::load_from_memory(&*blob.contents)?;
    let image_dimensions = image.dimensions();
    tracing::info!("Loaded image: {:?} {:?}", image_dimensions, image.color());

    let image = image.to_rgba();

    Ok(RawRgba8Image {
        data: image.into_raw(),
        dimensions: image_dimensions,
    })
}

fn load_ldr_tex(image: &RawRgba8Image, params: &TexParams) -> Result<Texture> {
    let internal_format = if params.gamma == TexGamma::Linear {
        vk::Format::R8G8B8A8_UNORM
    } else {
        vk::Format::R8G8B8A8_SRGB
    };

    load_tex_impl(&image.data, image.dimensions, internal_format)
}

fn load_tex_impl(
    image_data: &[u8],
    image_dimensions: (u32, u32),
    internal_format: vk::Format,
) -> Result<Texture> {
    use crate::vulkan::*;
    use ash::util::Align;
    use ash::version::DeviceV1_0;

    let device = vk_device();
    let image_buffer_info = vk::BufferCreateInfo {
        size: (std::mem::size_of::<u8>() * image_data.len()) as u64,
        usage: vk::BufferUsageFlags::TRANSFER_SRC,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };

    let buffer_mem_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::CpuToGpu,
        ..Default::default()
    };

    let (image_buffer, buffer_allocation, buffer_allocation_info) = unsafe { vk_all() }
        .allocator
        .create_buffer(&image_buffer_info, &buffer_mem_info)
        .expect("vma::create_buffer");

    unsafe {
        let image_ptr = vk_all()
            .allocator
            .map_memory(&buffer_allocation)
            .expect("mapping an image upload buffer failed")
            as *mut std::ffi::c_void;
        let mut image_slice = Align::new(
            image_ptr,
            std::mem::align_of::<u8>() as u64,
            buffer_allocation_info.get_size() as u64,
        );

        image_slice.copy_from_slice(image_data);
        vk_all()
            .allocator
            .unmap_memory(&buffer_allocation)
            .expect("unmap_memory");
    }

    let res = backend::texture::create_texture(TextureKey {
        width: image_dimensions.0,
        height: image_dimensions.1,
        format: internal_format.as_raw(),
    });

    let res_image = res.image;
    vk_add_setup_command(move |_vk_all, vk_frame| {
        vk_frame
            .frame_cleanup
            .lock()
            .unwrap()
            .push(Box::new(move |vk_all| {
                vk_all
                    .allocator
                    .destroy_buffer(image_buffer, &buffer_allocation)
                    .unwrap()
            }));

        let cb = vk_frame.command_buffer.lock().unwrap();
        let cb: vk::CommandBuffer = cb.cb;

        record_image_barrier(
            vk_device(),
            cb,
            ImageBarrier::new(
                res_image,
                vk_sync::AccessType::Nothing,
                vk_sync::AccessType::TransferWrite,
            )
            .with_discard(true),
        );

        let buffer_copy_regions = vk::BufferImageCopy::builder()
            .image_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .build(),
            )
            .image_extent(vk::Extent3D {
                width: image_dimensions.0,
                height: image_dimensions.1,
                depth: 1,
            });

        unsafe {
            device.cmd_copy_buffer_to_image(
                cb,
                image_buffer,
                res_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[buffer_copy_regions.build()],
            );

            record_image_barrier(
                vk_device(),
                cb,
                ImageBarrier::new(
                    res_image,
                    vk_sync::AccessType::TransferWrite,
                    vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
                ),
            )
        };
    });

    /*match img {
        DynamicImage::ImageLuma8(ref img) => make_gl_tex(img, dims, gl::R8, gl::RED),
        DynamicImage::ImageRgb8(ref img) => make_gl_tex(
            img,
            dims,
            if params.gamma == TexGamma::Linear {
                gl::RGB8
            } else {
                gl::SRGB8
            },
            gl::RGB,
        ),
        DynamicImage::ImageRgba8(ref img) => make_gl_tex(
            img,
            dims,
            if params.gamma == TexGamma::Linear {
                gl::RGBA8
            } else {
                gl::SRGB8_ALPHA8
            },
            gl::RGBA,
        ),
        _ => Err(format_err!("Unsupported image format")),
    }*/

    Ok(res)
}

fn load_hdr_tex(blob: &Blob, _params: &TexParams) -> Result<Texture> {
    let _img = hdrldr::load(blob.contents.as_slice()).map_err(|e| format_err!("{:?}", e))?;

    unimplemented!()
    /*
    // Flip the image because OpenGL.
    for y in 0..img.height / 2 {
        for x in 0..img.width {
            let y2 = img.height - 1 - y;
            img.data.swap(y * img.width + x, y2 * img.width + x);
        }
    }

    tracing::info!("Loaded image: {}x{} HDR", img.width, img.height);

    with_gl(|gl| {
        let res = backend::texture::create_texture(
            gl,
            TextureKey {
                width: img.width as u32,
                height: img.height as u32,
                format: gl::RGB32F,
            },
        );
        unsafe {
            gl.BindTexture(gl::TEXTURE_2D, res.texture_id);
            gl.TexSubImage2D(
                gl::TEXTURE_2D,
                0,
                0,
                0,
                img.width as i32,
                img.height as i32,
                gl::RGB,
                gl::FLOAT,
                std::mem::transmute(img.data.as_ptr()),
            );
        }
        Ok(res)
    })*/
}

#[snoozy]
pub async fn load_tex_with_params(
    ctx: Context,
    path: &AssetPath,
    params: &TexParams,
) -> Result<Texture> {
    if path.asset_name.ends_with(".hdr") {
        let blob = ctx.get(&load_blob(path.clone())).await?;
        load_hdr_tex(&*blob, params)
    } else {
        let raw_img = ctx.get(&load_raw_ldr_tex(path.clone())).await?;
        load_ldr_tex(&*raw_img, params)
    }
}

#[snoozy]
pub async fn make_placeholder_rgba8_tex(_ctx: Context, texel_value: &[u8; 4]) -> Result<Texture> {
    let image_dimensions = (1, 1);
    let internal_format = vk::Format::R8G8B8A8_UNORM;

    load_tex_impl(texel_value, image_dimensions, internal_format)
}
