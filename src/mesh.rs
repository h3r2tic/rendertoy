use crate::blob::Blob;
use snoozy::*;
use std::io;

pub struct Mesh {
    #[allow(dead_code)]
    models: Vec<tobj::Model>,
}

snoozy! {
    fn load_mesh(ctx: &mut Context, blob: &SnoozyRef<Blob>) -> Result<Mesh> {
        let m = tobj::load_obj_buf(&mut io::Cursor::new(&ctx.get(blob)?.contents), |_| {
            Ok((Default::default(), Default::default()))
        })?;

        let (models, _materials) = m;
        Ok(Mesh { models: models })
    }
}
