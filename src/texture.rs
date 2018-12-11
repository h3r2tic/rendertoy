pub use crate::backend::texture::{Texture, TextureKey};

use crate::backend;
use crate::blob::{load_blob, AssetPath};

use snoozy::*;

snoozy! {
    fn load_tex(ctx: &mut Context, path: &AssetPath) -> Result<Texture> {
        use image::{DynamicImage, GenericImageView};
		use image::imageops::flip_vertical;

        let blob = ctx.get(&load_blob(path.clone()))?;
        let img = image::load_from_memory(&blob.contents)?;

        let dims = img.dimensions();
        println!("Loaded image: {:?} {:?}", dims, img.color());

        let res = backend::texture::create_texture(TextureKey{ width: dims.0, height: dims.1, format: gl::SRGB8_ALPHA8 });

        unsafe { gl::BindTexture(gl::TEXTURE_2D, res.texture_id); }

        match img {
            DynamicImage::ImageRgb8(ref img) => unsafe {
				let img_flipped = flip_vertical(img);
				gl::TexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, dims.0 as i32, dims.1 as i32, gl::RGB, gl::UNSIGNED_BYTE, std::mem::transmute(img_flipped.into_raw().as_ptr()));
			},
            DynamicImage::ImageRgba8(ref img) => unsafe {
				let img_flipped = flip_vertical(img);
				gl::TexSubImage2D(gl::TEXTURE_2D, 0, 0, 0, dims.0 as i32, dims.1 as i32, gl::RGBA, gl::UNSIGNED_BYTE, std::mem::transmute(img_flipped.into_raw().as_ptr()));
			},
            _ => return Err(format_err!("Unsupported image format"))
        }

        Ok(res)
    }
}
