pub use crate::backend::texture::{Texture, TextureKey};

use crate::backend;
use crate::blob::{load_blob, AssetPath};

use snoozy::*;

#[derive(Serialize, Debug, PartialEq, Eq)]
pub enum TexGamma {
    Linear,
    Srgb,
}

#[derive(Serialize, Debug)]
pub struct TexParams {
    pub gamma: TexGamma,
}

snoozy! {
    fn load_tex(ctx: &mut Context, path: &AssetPath) -> Result<Texture> {
        let tex = ctx.get(load_tex_with_params(path.clone(), TexParams { gamma: TexGamma::Srgb }))?;
        Ok((*tex).clone())
    }
}

snoozy! {
    fn load_tex_with_params(ctx: &mut Context, path: &AssetPath, params: &TexParams) -> Result<Texture> {
        use image::{DynamicImage, GenericImageView};
        use image::imageops::flip_vertical;

        let blob = ctx.get(&load_blob(path.clone()))?;
        let img = image::load_from_memory(&blob.contents)?;

        let dims = img.dimensions();
        println!("Loaded image: {:?} {:?}", dims, img.color());

        match img {
            DynamicImage::ImageLuma8(ref img) => unsafe {
                let img_flipped = flip_vertical(img);
                let res = backend::texture::create_texture(TextureKey{ width: dims.0, height: dims.1, format: gl::R8 });
                gl::BindTexture(gl::TEXTURE_2D, res.texture_id);
                gl::TexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, dims.0 as i32, dims.1 as i32, gl::RED, gl::UNSIGNED_BYTE, std::mem::transmute(img_flipped.into_raw().as_ptr()));
                Ok(res)
            },
            DynamicImage::ImageRgb8(ref img) => unsafe {
                let img_flipped = flip_vertical(img);
                let res = backend::texture::create_texture(TextureKey{ width: dims.0, height: dims.1, format: if params.gamma == TexGamma::Linear { gl::RGB8 } else { gl::SRGB8 } });
                gl::BindTexture(gl::TEXTURE_2D, res.texture_id);
                gl::TexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, dims.0 as i32, dims.1 as i32, gl::RGB, gl::UNSIGNED_BYTE, std::mem::transmute(img_flipped.into_raw().as_ptr()));
                Ok(res)
            },
            DynamicImage::ImageRgba8(ref img) => unsafe {
                let img_flipped = flip_vertical(img);
                let res = backend::texture::create_texture(TextureKey{ width: dims.0, height: dims.1, format: if params.gamma == TexGamma::Linear { gl::RGBA8 } else { gl::SRGB8_ALPHA8 } });
                gl::BindTexture(gl::TEXTURE_2D, res.texture_id);
                gl::TexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, dims.0 as i32, dims.1 as i32, gl::RGBA, gl::UNSIGNED_BYTE, std::mem::transmute(img_flipped.into_raw().as_ptr()));
                Ok(res)
            },
            _ => Err(format_err!("Unsupported image format"))
        }
    }
}
