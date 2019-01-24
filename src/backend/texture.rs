use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Eq, PartialEq, Hash, Clone, Copy, Serialize, Debug)]
pub struct TextureKey {
    pub width: u32,
    pub height: u32,
    pub format: u32,
}

#[derive(Clone)]
pub struct Texture {
    pub texture_id: u32,
    pub sampler_id: u32,
    pub key: TextureKey,
    _allocation: std::sync::Arc<TextureAllocation>,
}

pub fn create_texture(key: TextureKey) -> Texture {
    let alloc = TextureAllocation::new(key);
    Texture {
        texture_id: alloc.texture_id,
        sampler_id: alloc.sampler_id,
        key,
        _allocation: std::sync::Arc::new(alloc),
    }
}

#[derive(Clone)]
struct TextureAllocation {
    key: TextureKey,
    texture_id: u32,
    sampler_id: u32,
}

lazy_static! {
    static ref TRANSIENT_TEXTURE_CACHE: Mutex<HashMap<TextureKey, Vec<TextureAllocation>>> =
        { Mutex::new(HashMap::new()) };
}

impl Drop for TextureAllocation {
    fn drop(&mut self) {
        TRANSIENT_TEXTURE_CACHE
            .lock()
            .unwrap()
            .entry(self.key)
            .or_default()
            .push(self.clone());
    }
}

impl TextureAllocation {
    fn new(key: TextureKey) -> TextureAllocation {
        let mut texcache_lock = TRANSIENT_TEXTURE_CACHE.lock().unwrap();
        let existing = texcache_lock.entry(key).or_default();

        if existing.is_empty() {
            //println!("Allocating new texture: {:?}", key);

            unsafe {
                let mut texture_id = 0;
                gl::GenTextures(1, &mut texture_id);
                gl::BindTexture(gl::TEXTURE_2D, texture_id);
                gl::TexStorage2D(
                    gl::TEXTURE_2D,
                    1,
                    key.format,
                    key.width as i32,
                    key.height as i32,
                );
                gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
                gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

                let mut sampler_id = 0;
                gl::GenSamplers(1, &mut sampler_id);
                gl::SamplerParameteri(sampler_id, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
                gl::SamplerParameteri(sampler_id, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

                TextureAllocation {
                    key,
                    texture_id,
                    sampler_id,
                }
            }
        } else {
            existing.pop().unwrap()
        }
    }
}
