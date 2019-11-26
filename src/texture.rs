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

#[derive(Serialize, Debug, Clone, Abomonation)]
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

fn make_gl_tex<Img>(
    img: &Img,
    dims: (u32, u32),
    internal_format: u32,
    layout: u32,
) -> Result<Texture>
where
    Img: image::GenericImageView + 'static,
{
    let img_flipped = image::imageops::flip_vertical(img);

    unimplemented!()
    /*with_gl(|gl| {
        let res = backend::texture::create_texture(
            gl,
            TextureKey {
                width: dims.0,
                height: dims.1,
                format: internal_format,
            },
        );
        unsafe {
            gl.BindTexture(gl::TEXTURE_2D, res.texture_id);
            gl.TexSubImage2D(
                gl::TEXTURE_2D,
                0,
                0,
                0,
                dims.0 as i32,
                dims.1 as i32,
                layout,
                gl::UNSIGNED_BYTE,
                std::mem::transmute(img_flipped.into_raw().as_ptr()),
            );
        }
        Ok(res)
    })*/
}

fn load_ldr_tex(blob: &Blob, params: &TexParams) -> Result<Texture> {
    use crate::vulkan::*;
    use ash::util::Align;
    use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0, InstanceV1_1};
    use image::{DynamicImage, GenericImageView, ImageBuffer};

    let device = vk_device();

    let image = image::load_from_memory(&blob.contents)?;
    let image_dimensions = image.dimensions();
    println!("Loaded image: {:?} {:?}", image_dimensions, image.color());

    // TODO: don't
    let image = image.to_rgba();
    let internal_format = if params.gamma == TexGamma::Linear {
        vk::Format::R8G8B8A8_UNORM
    } else {
        vk::Format::R8G8B8A8_SRGB
    };

    let image_data = image.into_raw();
    let image_buffer_info = vk::BufferCreateInfo {
        size: (std::mem::size_of::<u8>() * image_data.len()) as u64,
        usage: vk::BufferUsageFlags::TRANSFER_SRC,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let image_buffer = unsafe { device.create_buffer(&image_buffer_info, None).unwrap() };
    let image_buffer_memory_req = unsafe { device.get_buffer_memory_requirements(image_buffer) };
    let image_buffer_memory_index = find_memorytype_index(
        &image_buffer_memory_req,
        &unsafe { vk_all().device_memory_properties },
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .expect("Unable to find suitable memorytype for the vertex buffer.");

    let image_buffer_allocate_info = vk::MemoryAllocateInfo {
        allocation_size: image_buffer_memory_req.size,
        memory_type_index: image_buffer_memory_index,
        ..Default::default()
    };

    unsafe {
        let image_buffer_memory = device
            .allocate_memory(&image_buffer_allocate_info, None)
            .unwrap();
        let image_ptr = unsafe {
            device.map_memory(
                image_buffer_memory,
                0,
                image_buffer_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .unwrap();
        let mut image_slice = Align::new(
            image_ptr,
            std::mem::align_of::<u8>() as u64,
            image_buffer_memory_req.size,
        );

        image_slice.copy_from_slice(&image_data);
        device.unmap_memory(image_buffer_memory);
        device
            .bind_buffer_memory(image_buffer, image_buffer_memory, 0)
            .unwrap();
    }

    let res = backend::texture::create_texture(TextureKey {
        width: image_dimensions.0,
        height: image_dimensions.1,
        format: internal_format.as_raw(),
    });

    let res_image = res.image;
    vk_add_setup_command(move |vk_all, vk_frame| {
        let cb = vk_frame.command_buffer.lock().unwrap();
        let cb: vk::CommandBuffer = cb.cb;

        unsafe {
            vk_all.record_image_barrier(
                cb,
                ImageBarrier::new(
                    res_image,
                    vk_sync::AccessType::Nothing,
                    vk_sync::AccessType::TransferWrite,
                )
                .with_discard(true),
            )
        };

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

            vk_all.record_image_barrier(
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
    let mut img = hdrldr::load(blob.contents.as_slice()).map_err(|e| format_err!("{:?}", e))?;

    unimplemented!()
    /*
    // Flip the image because OpenGL.
    for y in 0..img.height / 2 {
        for x in 0..img.width {
            let y2 = img.height - 1 - y;
            img.data.swap(y * img.width + x, y2 * img.width + x);
        }
    }

    println!("Loaded image: {}x{} HDR", img.width, img.height);

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
    let blob = ctx.get(&load_blob(path.clone())).await?;

    if path.asset_name.ends_with(".hdr") {
        load_hdr_tex(&*blob, params)
    } else {
        load_ldr_tex(&*blob, params)
    }
}

#[snoozy]
pub async fn make_placeholder_rgba8_tex(_ctx: Context, texel_value: &[u8; 4]) -> Result<Texture> {
    unimplemented!()

    /*with_gl(|gl| unsafe {
        let res = backend::texture::create_texture(
            gl,
            TextureKey {
                width: 1,
                height: 1,
                format: gl::RGBA8,
            },
        );

        gl.BindTexture(gl::TEXTURE_2D, res.texture_id);
        gl.TexSubImage2D(
            gl::TEXTURE_2D,
            0,
            0,
            0,
            1,
            1,
            gl::RGBA,
            gl::UNSIGNED_BYTE,
            std::mem::transmute(texel_value.as_ptr()),
        );
        Ok(res)
    })*/
}
