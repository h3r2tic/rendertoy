use super::transient_resource::*;

#[derive(Eq, PartialEq, Hash, Clone, Copy, Serialize, Debug)]
pub struct RenderBufferKey {
    pub width: u32,
    pub height: u32,
    pub format: u32,
}

#[derive(Clone)]
pub struct RenderBuffer {
    pub render_buffer_id: u32,
    pub key: RenderBufferKey,
    _allocation: SharedTransientAllocation,
}

#[derive(Clone)]
pub struct RenderBufferAllocation {
    render_buffer_id: u32,
}

pub fn create_render_buffer(gl: &gl::Gl, key: RenderBufferKey) -> RenderBuffer {
    create_transient(gl, key)
}

impl TransientResource for RenderBuffer {
    type Desc = RenderBufferKey;
    type Allocation = RenderBufferAllocation;

    fn new(
        desc: RenderBufferKey,
        allocation: std::sync::Arc<
            TransientResourceAllocation<RenderBufferKey, RenderBufferAllocation>,
        >,
    ) -> Self {
        Self {
            render_buffer_id: allocation.payload.render_buffer_id,
            key: desc,
            _allocation: allocation,
        }
    }

    fn allocate_payload(gl: &gl::Gl, key: RenderBufferKey) -> RenderBufferAllocation {
        unsafe {
            let mut render_buffer_id = 0;
            gl.GenRenderbuffers(1, &mut render_buffer_id);
            gl.BindRenderbuffer(gl::RENDERBUFFER, render_buffer_id);
            gl.RenderbufferStorage(
                gl::RENDERBUFFER,
                key.format,
                key.width as i32,
                key.height as i32,
            );

            RenderBufferAllocation { render_buffer_id }
        }
    }
}
