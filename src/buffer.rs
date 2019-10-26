pub use crate::backend::buffer::{Buffer, BufferKey};

use crate::backend;

use snoozy::*;
use std::mem::size_of;

pub fn upload_buffer_impl<T: Copy + Send + 'static>(
    _ctx: &mut Context,
    contents: &T,
) -> Result<Buffer> {
    let size_of_t = size_of::<T>();

    let res = backend::buffer::create_buffer(BufferKey {
        size_bytes: size_of_t,
        texture_format: None,
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
pub fn upload_buffer<T: Copy + Send + 'static>(ctx: &mut Context, contents: &T) -> Result<Buffer> {
    upload_buffer_impl(ctx, contents)
}

use std::ops::Deref;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::Arc;

pub struct ArcView<T, OwnerT> {
    owner: Pin<Arc<OwnerT>>,
    child: NonNull<T>,
}

unsafe impl<T, OwnerT> Send for ArcView<T, OwnerT> {}
unsafe impl<T, OwnerT> Sync for ArcView<T, OwnerT> {}

impl<T: 'static, OwnerT> ArcView<T, OwnerT> {
    pub fn new<'a, F>(owner: &'a Pin<Arc<OwnerT>>, get_member: F) -> Self
    where
        F: for<'b> FnOnce(&'b Pin<Arc<OwnerT>>) -> &'b T,
    {
        let owner = owner.clone();
        let child = NonNull::new(get_member(&owner) as *const T as *mut T).unwrap();

        Self { owner, child }
    }
}

impl<T: 'static, OwnerT> Clone for ArcView<T, OwnerT> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            child: self.child,
        }
    }
}

impl<T: 'static, OwnerT> Deref for ArcView<T, OwnerT> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { self.child.as_ref() }
    }
}

impl<T: 'static, OwnerT> std::hash::Hash for ArcView<T, OwnerT> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        whatever_hash(unsafe { self.child.as_ref() }, state);
    }
}

pub fn upload_array_buffer_impl<
    T: Sized + 'static,
    C: Deref<Target = Vec<T>> + Send + Sync + Sized + 'static,
>(
    _ctx: &mut Context,
    contents: &C,
    texture_format: Option<u32>,
) -> Result<Buffer> {
    let size_of_t = size_of::<T>();

    let res = backend::buffer::create_buffer(BufferKey {
        size_bytes: contents.len() * size_of_t,
        texture_format,
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

#[snoozy]
pub fn upload_array_buffer<
    T: Sized + 'static,
    C: Deref<Target = Vec<T>> + Send + Sync + Sized + 'static,
>(
    ctx: &mut Context,
    contents: &C,
) -> Result<Buffer> {
    upload_array_buffer_impl(ctx, contents, None)
}

#[snoozy]
pub fn upload_array_tex_buffer<
    T: Sized + 'static,
    C: Deref<Target = Vec<T>> + Send + Sync + Sized + 'static,
>(
    ctx: &mut Context,
    contents: &C,
    texture_format: &u32,
) -> Result<Buffer> {
    upload_array_buffer_impl(ctx, contents, Some(*texture_format))
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
