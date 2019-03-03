pub use crate::backend::texture::{Texture, TextureKey};

use crate::backend;
use crate::blob::{load_blob, AssetPath};

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
pub fn load_tex(ctx: &mut Context, path: &AssetPath) -> Result<Texture> {
    let tex = ctx.get(load_tex_with_params(
        path.clone(),
        TexParams {
            gamma: TexGamma::Srgb,
        },
    ))?;
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

#[snoozy]
pub fn load_tex_with_params(
    ctx: &mut Context,
    path: &AssetPath,
    params: &TexParams,
) -> Result<Texture> {
    use image::{DynamicImage, GenericImageView};

    let blob = ctx.get(&load_blob(path.clone()))?;
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

#[snoozy]
pub fn make_placeholder_rgba8_tex(_ctx: &mut Context, texel_value: &[u8; 4]) -> Result<Texture> {
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
