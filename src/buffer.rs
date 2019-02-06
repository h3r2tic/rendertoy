pub use crate::backend::buffer::{Buffer, BufferKey};

use crate::backend;

use snoozy::*;

snoozy! {
    fn upload_buffer(_ctx: &mut Context, contents: &Vec<u8>) -> Result<Buffer> {
        let res = backend::buffer::create_buffer(BufferKey {
            size_bytes: contents.len(),
        });

        unsafe {
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, res.buffer_id);
            gl::BufferSubData(
                gl::SHADER_STORAGE_BUFFER,
                0,
                contents.len() as isize,
                contents.as_ptr() as *const std::ffi::c_void,
            );
        }

        Ok(res)
    }
}

pub fn to_byte_vec<T>(mut v: Vec<T>) -> Vec<u8>
where
    T: Copy,
{
    unsafe {
        let p = v.as_mut_ptr();
        let item_sizeof = std::mem::size_of::<T>();
        let len = v.len() * item_sizeof;
        let cap = v.capacity() * item_sizeof;
        std::mem::forget(v);
        Vec::from_raw_parts(p as *mut u8, len, cap)
    }
}
