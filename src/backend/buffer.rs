use super::transient_resource::*;

#[derive(Eq, PartialEq, Hash, Clone, Copy, Serialize, Debug)]
pub struct BufferKey {
    pub size_bytes: usize,
    pub texture_format: Option<u32>,
}

#[derive(Clone)]
pub struct BufferAllocation {
    buffer_id: u32,
    texture_id: Option<u32>,
    bindless_texture_handle: Option<u64>,
}

#[derive(Clone)]
pub struct Buffer {
    pub buffer_id: u32,
    pub texture_id: Option<u32>,
    pub bindless_texture_handle: Option<u64>,
    pub key: BufferKey,
    _allocation: SharedTransientAllocation,
}

impl TransientResource for Buffer {
    type Desc = BufferKey;
    type Allocation = BufferAllocation;

    fn new(
        desc: BufferKey,
        allocation: std::sync::Arc<TransientResourceAllocation<BufferKey, BufferAllocation>>,
    ) -> Self {
        Self {
            buffer_id: allocation.payload.buffer_id,
            texture_id: allocation.payload.texture_id,
            bindless_texture_handle: allocation.payload.bindless_texture_handle,
            key: desc,
            _allocation: allocation,
        }
    }

    fn allocate_payload(gfx: &crate::Gfx, key: BufferKey) -> BufferAllocation {
        /*unsafe {
            let mut buffer_id = 0;
            gl.GenBuffers(1, &mut buffer_id);
            gl.BindBuffer(gl::SHADER_STORAGE_BUFFER, buffer_id);

            gl.BufferStorage(
                gl::SHADER_STORAGE_BUFFER,
                key.size_bytes as isize,
                std::ptr::null(),
                gl::DYNAMIC_STORAGE_BIT,
            );

            let tex = key.texture_format.map(|internal_format| {
                let mut texture_id = 0u32;
                gl.GenTextures(1, &mut texture_id);
                gl.BindTexture(gl::TEXTURE_BUFFER, texture_id);
                gl.TexBuffer(gl::TEXTURE_BUFFER, internal_format, buffer_id);
                gl.BindTexture(gl::TEXTURE_BUFFER, 0);

                let bindless_texture_handle: u64 = gl.GetTextureHandleARB(texture_id);
                gl.MakeTextureHandleResidentARB(bindless_texture_handle);

                (texture_id, bindless_texture_handle)
            });

            BufferAllocation {
                buffer_id,
                texture_id: tex.map(|t| t.0),
                bindless_texture_handle: tex.map(|t| t.1),
            }
        }*/
        unimplemented!()
    }
}

pub fn create_buffer(gfx: &crate::Gfx, key: BufferKey) -> Buffer {
    create_transient(gfx, key)
}
