use snoozy::*;
use std::fs::File;

#[derive(Hash, Debug)]
pub struct Blob {
    pub contents: Vec<u8>,
}

snoozy! {
    fn load_file(ctx: &mut Context, path: &String) -> Result<Blob> {
        let mut buffer = Vec::new();
        std::io::Read::read_to_end(&mut File::open(&path)?, &mut buffer)?;

        crate::backend::file::watch_file(&path, ctx.get_invalidation_trigger());

        Ok(Blob { contents: buffer })
    }
}
