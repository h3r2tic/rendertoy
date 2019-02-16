pub use crate::backend::buffer::{Buffer, BufferKey};

use crate::backend;

use snoozy::*;
use std::mem::size_of;

#[snoozy]
pub fn upload_buffer<T: Copy + Send + 'static>(_ctx: &mut Context, contents: &T) -> Result<Buffer> {
    let size_of_t = size_of::<T>();

    let res = backend::buffer::create_buffer(BufferKey {
        size_bytes: size_of_t,
    });

    unsafe {
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, res.buffer_id);
        gl::BufferSubData(
            gl::SHADER_STORAGE_BUFFER,
            0,
            size_of_t as isize,
            std::mem::transmute(contents),
        );
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
    }

    Ok(res)
}

#[snoozy]
pub fn upload_array_buffer<T: Copy + Send + 'static>(
    _ctx: &mut Context,
    contents: &Vec<T>,
) -> Result<Buffer> {
    let size_of_t = size_of::<T>();

    let res = backend::buffer::create_buffer(BufferKey {
        size_bytes: contents.len() * size_of_t,
    });

    unsafe {
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, res.buffer_id);
        gl::BufferSubData(
            gl::SHADER_STORAGE_BUFFER,
            0,
            (contents.len() * size_of_t) as isize,
            contents.as_ptr() as *const T as *const std::ffi::c_void,
        );
        gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
    }

    Ok(res)
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
