pub use crate::backend::texture::{Texture, TextureKey};

use crate::backend;
use crate::blob::{load_blob, AssetPath};

use snoozy::*;

snoozy! {
    fn load_tex(ctx: &mut Context, path: &AssetPath) -> Result<Texture> {
        use image::GenericImageView;

        let blob = ctx.get(&load_blob(path.clone()))?;
        let img = image::load_from_memory(&blob.contents)?;

        let dims = img.dimensions();
        println!("Loaded image: {:?} {:?}", dims, img.color());

        // TODO: upload pixels
        Ok(backend::texture::create_texture(TextureKey{ width: dims.0, height: dims.1, format: gl::SRGB8_ALPHA8 }))
    }
}
