pub use crate::backend::texture::{Texture, TextureKey};

use crate::backend;
use crate::blob::load_file;

use snoozy::*;

snoozy! {
    fn load_tex(ctx: &mut Context, path: &String) -> Result<Texture> {
        use image::GenericImageView;

        let blob = ctx.get(&load_file("assets/tex/".to_string() + path))?;
        let img = image::load_from_memory(&blob.contents)?;

        let dims = img.dimensions();
        println!("Loaded image: {:?} {:?}", dims, img.color());

        // TODO: upload pixels
        Ok(backend::texture::create_texture(TextureKey{ width: dims.0, height: dims.1, format: gl::SRGB8_ALPHA8 }))
    }
}
