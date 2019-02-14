use super::transient_resource::*;

#[derive(Eq, PartialEq, Hash, Clone, Copy, Serialize, Debug)]
pub struct BufferKey {
    pub size_bytes: usize,
}

#[derive(Clone)]
pub struct BufferAllocation {
    buffer_id: u32,
}

#[derive(Clone)]
pub struct Buffer {
    pub buffer_id: u32,
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
            key: desc,
            _allocation: allocation,
        }
    }

    fn allocate_payload(key: BufferKey) -> BufferAllocation {
        unsafe {
            let mut buffer_id = 0;
            gl::GenBuffers(1, &mut buffer_id);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, buffer_id);

            gl::BufferStorage(
                gl::SHADER_STORAGE_BUFFER,
                key.size_bytes as isize,
                std::ptr::null(),
                gl::DYNAMIC_STORAGE_BIT,
            );

            BufferAllocation { buffer_id }
        }
    }
}

pub fn create_buffer(key: BufferKey) -> Buffer {
    create_transient(key)
}
