pub use crate::backend::buffer::{Buffer, BufferKey};
use crate::backend::{self};
use crate::{vk, vulkan::*};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0, InstanceV1_1};

use snoozy::*;
use std::mem::size_of;

fn upload_buffer_impl<T: Copy + Send + Sync + 'static>(
    gfx: &crate::Gfx,
    _ctx: Context,
    contents: &T,
) -> Result<Buffer> {
    /*let size_of_t = size_of::<T>();

    let res = backend::buffer::create_buffer(
        gl,
        BufferKey {
            size_bytes: size_of_t,
            texture_format: None,
        },
    );

    unsafe {
        gl.BindBuffer(gl::SHADER_STORAGE_BUFFER, res.buffer_id);
        gl.BufferSubData(
            gl::SHADER_STORAGE_BUFFER,
            0,
            size_of_t as isize,
            std::mem::transmute(contents),
        );
        gl.BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
    }

    Ok(res)*/
    unimplemented!()
}

#[snoozy]
pub async fn upload_buffer<T: Copy + Send + Sync + 'static>(
    ctx: Context,
    contents: &T,
) -> Result<Buffer> {
    //with_gl(|gl| upload_buffer_impl(gfx, ctx, contents))
    unimplemented!()
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
    _ctx: Context,
    contents: &C,
    texture_format: Option<vk::Format>,
) -> Result<Buffer> {
    let size_of_t = size_of::<T>();

    let size_bytes = contents.len() * size_of_t;
    let res = backend::buffer::create_buffer(BufferKey::new(size_bytes, texture_format));

    // TODO
    let (staging_buffer, staging_allocation, staging_allocation_info) = unsafe {
        let usage: vk::BufferUsageFlags = vk::BufferUsageFlags::TRANSFER_SRC;

        let mem_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::CpuToGpu,
            ..Default::default()
        };

        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size_bytes as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        vk_all()
            .allocator
            .create_buffer(&buffer_info, &mem_info)
            .expect("vma::create_buffer")
    };

    //(contents.len() * size_of_t) as isize,
    //contents.as_ptr() as *const T as *const std::ffi::c_void,

    unsafe {
        let mapped_ptr = vk_all()
            .allocator
            .map_memory(&staging_allocation)
            .expect("mapping a staging buffer failed")
            as *mut std::ffi::c_void;

        std::slice::from_raw_parts_mut(mapped_ptr as *mut u8, size_bytes).copy_from_slice(
            &std::slice::from_raw_parts(contents.as_ptr() as *const u8, size_bytes),
        );
        vk_all().allocator.unmap_memory(&staging_allocation);
    }

    let copy_dst_buffer = res.buffer;
    vk_add_setup_command(move |vk_all, vk_frame| {
        let cb = vk_frame.command_buffer.lock().unwrap();
        let cb: vk::CommandBuffer = cb.cb;

        let buffer_copy_regions = vk::BufferCopy::builder().size(size_bytes as u64);

        unsafe {
            vk_all.device.cmd_copy_buffer(
                cb,
                staging_buffer,
                copy_dst_buffer,
                &[buffer_copy_regions.build()],
            );
        };

        // TODO: barrier
    });

    // TODO: free the staging buffer

    Ok(res)
}

#[snoozy]
pub async fn upload_array_buffer<
    T: Sized + 'static,
    C: Deref<Target = Vec<T>> + Send + Sync + Sized + 'static,
>(
    ctx: Context,
    contents: &C,
) -> Result<Buffer> {
    upload_array_buffer_impl(ctx, contents, None)
}

#[snoozy]
pub async fn upload_array_tex_buffer<
    T: Sized + 'static,
    C: Deref<Target = Vec<T>> + Send + Sync + Sized + 'static,
>(
    ctx: Context,
    contents: &C,
    texture_format: &vk::Format,
) -> Result<Buffer> {
    //with_gl(|gl| upload_array_buffer_impl(gfx, ctx, contents, Some(*texture_format)))
    unimplemented!()
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
