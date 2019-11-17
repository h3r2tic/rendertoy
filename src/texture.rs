pub use crate::backend::texture::{Texture, TextureKey};

use crate::backend;
use crate::blob::{load_blob, AssetPath, Blob};

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
    let res = backend::texture::create_texture(TextureKey {
        width: dims.0,
        height: dims.1,
        format: internal_format,
    });
    unsafe {
        gl::BindTexture(gl::TEXTURE_2D, res.texture_id);
        gl::TexSubImage2D(
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
}

fn load_ldr_tex(blob: &Blob, params: &TexParams) -> Result<Texture> {
    use image::{DynamicImage, GenericImageView};

    let img = image::load_from_memory(&blob.contents)?;

    let dims = img.dimensions();
    println!("Loaded image: {:?} {:?}", dims, img.color());

    match img {
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
    }
}

fn load_hdr_tex(blob: &Blob, _params: &TexParams) -> Result<Texture> {
    let mut img = hdrldr::load(blob.contents.as_slice()).map_err(|e| format_err!("{:?}", e))?;

    // Flip the image because OpenGL.
    for y in 0..img.height / 2 {
        for x in 0..img.width {
            let y2 = img.height - 1 - y;
            img.data.swap(y * img.width + x, y2 * img.width + x);
        }
    }

    println!("Loaded image: {}x{} HDR", img.width, img.height);

    let res = backend::texture::create_texture(TextureKey {
        width: img.width as u32,
        height: img.height as u32,
        format: gl::RGB32F,
    });
    unsafe {
        gl::BindTexture(gl::TEXTURE_2D, res.texture_id);
        gl::TexSubImage2D(
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
    let res = backend::texture::create_texture(TextureKey {
        width: 1,
        height: 1,
        format: gl::RGBA8,
    });
    unsafe {
        gl::BindTexture(gl::TEXTURE_2D, res.texture_id);
        gl::TexSubImage2D(
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
    }
    Ok(res)
}
