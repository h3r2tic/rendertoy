use super::transient_resource::*;
use crate::vk;

#[derive(Eq, PartialEq, Hash, Clone, Copy, Serialize, Debug)]
pub struct TextureKey {
    pub width: u32,
    pub height: u32,
    pub format: i32,
}

impl TextureKey {
    pub fn new(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
            width,
            height,
            format: format.as_raw(),
        }
    }

    pub fn res_div_round_up(&self, x: u32, y: u32) -> Self {
        let mut res = self.clone();
        res.width = (res.width + x - 1) / x;
        res.height = (res.height + y - 1) / y;
        res
    }

    pub fn padded(&self, x: u32, y: u32) -> Self {
        let mut res = self.clone();
        res.width += x;
        res.height += y;
        res
    }

    pub fn half_res(&self) -> Self {
        self.res_div_round_up(2, 2)
    }

    pub fn with_width(&self, v: u32) -> Self {
        let mut res = self.clone();
        res.width = v;
        res
    }

    pub fn with_height(&self, v: u32) -> Self {
        let mut res = self.clone();
        res.height = v;
        res
    }

    pub fn with_format(&self, format: vk::Format) -> Self {
        let mut res = self.clone();
        res.format = format.as_raw();
        res
    }
}

#[derive(Clone)]
pub struct Texture {
    pub texture_id: u32,
    pub sampler_id: u32,
    pub bindless_handle: u64,
    pub key: TextureKey,
    _allocation: SharedTransientAllocation,
}

#[derive(Clone)]
pub struct TextureAllocation {
    texture_id: u32,
    sampler_id: u32,
    bindless_handle: u64,
}

pub fn create_texture(gfx: &crate::Gfx, key: TextureKey) -> Texture {
    create_transient(gfx, key)
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
            bindless_handle: allocation.payload.bindless_handle,
            key: desc,
            _allocation: allocation,
        }
    }

    fn allocate_payload(gfx: &crate::Gfx, key: TextureKey) -> TextureAllocation {
        /*unsafe {
            let mut prev_bound_texture = 0;
            gl.GetIntegerv(gl::TEXTURE_BINDING_2D, &mut prev_bound_texture);

            let mut texture_id = 0;
            gl.GenTextures(1, &mut texture_id);
            gl.BindTexture(gl::TEXTURE_2D, texture_id);
            gl.TexStorage2D(
                gl::TEXTURE_2D,
                1,
                key.format,
                key.width as i32,
                key.height as i32,
            );
            gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
            gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
            gl.TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);

            let mut sampler_id = 0;
            gl.GenSamplers(1, &mut sampler_id);
            gl.SamplerParameteri(sampler_id, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl.SamplerParameteri(sampler_id, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
            gl.SamplerParameteri(sampler_id, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
            gl.SamplerParameteri(sampler_id, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);

            let bindless_handle = gl.GetTextureHandleARB(texture_id);
            gl.MakeTextureHandleResidentARB(bindless_handle);

            // Restore the previously bound texture
            gl.BindTexture(gl::TEXTURE_2D, prev_bound_texture as u32);

            TextureAllocation {
                texture_id,
                sampler_id,
                bindless_handle,
            }
        }*/
        unimplemented!()
    }
}
