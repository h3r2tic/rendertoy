use super::transient_resource::*;

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
    _allocation: SharedTransientAllocation,
}

#[derive(Clone)]
pub struct TextureAllocation {
    texture_id: u32,
    sampler_id: u32,
}

pub fn create_texture(key: TextureKey) -> Texture {
    create_transient(key)
}

impl TransientResource for Texture {
    type Desc = TextureKey;
    type Allocation = TextureAllocation;

    fn new(
        desc: TextureKey,
        allocation: std::sync::Arc<TransientResourceAllocation<TextureKey, TextureAllocation>>,
    ) -> Self {
        Self {
            texture_id: allocation.payload.texture_id,
            sampler_id: allocation.payload.sampler_id,
            key: desc,
            _allocation: allocation,
        }
    }

    fn allocate_payload(key: TextureKey) -> TextureAllocation {
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
                texture_id,
                sampler_id,
            }
        }
    }
}
