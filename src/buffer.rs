pub use crate::backend::buffer::{Buffer, BufferKey};
use crate::backend::{self};
use crate::{vk, vulkan::*};
use ash::version::DeviceV1_0;

use snoozy::*;
use std::mem::size_of;

#[snoozy]
pub async fn upload_buffer<T: Sized + Copy + Send + Sync + 'static>(
    _ctx: Context,
    contents: &T,
) -> Result<Buffer> {
    let r: &T = &*contents;
    let s: &[T] = std::slice::from_ref(r);
    upload_array_buffer_impl(&&s, None)
}

use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::Arc;

pub struct ArcView<T, OwnerT> {
    owner: Arc<OwnerT>,
    child: NonNull<T>,
}

unsafe impl<T, OwnerT> Send for ArcView<T, OwnerT> {}
unsafe impl<T, OwnerT> Sync for ArcView<T, OwnerT> {}

impl<T: 'static, OwnerT> ArcView<T, OwnerT> {
    pub fn new<'a, F>(owner: &'a Arc<OwnerT>, get_member: F) -> Self
    where
        F: for<'b> FnOnce(&'b Arc<OwnerT>) -> &'b T,
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

// Allows smart pointers to containers implementing AsRef<[T]>
pub fn upload_array_buffer_impl<T, CRef, TCont>(
    contents: &CRef,
    texture_format: Option<vk::Format>,
) -> Result<Buffer>
where
    T: Sized + Copy + 'static,
    CRef: std::ops::Deref<Target = TCont>,
    TCont: AsRef<[T]>,
{
    let contents: &[T] = AsRef::<[T]>::as_ref(&**contents);
    let size_of_t = size_of::<T>();

    let size_bytes = contents.len() * size_of_t;
    let res = backend::buffer::create_buffer(BufferKey::new(size_bytes, texture_format));

    let (staging_buffer, staging_allocation, _staging_allocation_info) = unsafe {
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

    unsafe {
        let mapped_ptr = vk_all()
            .allocator
            .map_memory(&staging_allocation)
            .expect("mapping a staging buffer failed")
            as *mut std::ffi::c_void;

        std::slice::from_raw_parts_mut(mapped_ptr as *mut u8, size_bytes).copy_from_slice(
            &std::slice::from_raw_parts(contents.as_ptr() as *const u8, size_bytes),
        );
        vk_all()
            .allocator
            .unmap_memory(&staging_allocation)
            .expect("unmap_memory");
    }

    let copy_dst_buffer = res.buffer;
    vk_add_setup_command(move |vk_all, vk_frame| {
        vk_frame
            .frame_cleanup
            .lock()
            .unwrap()
            .push(Box::new(move |vk_all| {
                vk_all
                    .allocator
                    .destroy_buffer(staging_buffer, &staging_allocation)
                    .unwrap()
            }));

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

        {
            let global_barrier = vk_sync::GlobalBarrier {
                previous_accesses: &[vk_sync::AccessType::TransferWrite],
                next_accesses: &[
                    vk_sync::AccessType::AnyShaderReadUniformBuffer,
                    vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
                ],
            };

            vk_sync::cmd::pipeline_barrier(
                vk_all.device.fp_v1_0(),
                cb,
                Some(global_barrier),
                &[],
                &[],
            );
        }
    });

    Ok(res)
}

#[snoozy]
pub async fn upload_array_buffer<
    T: Sized + Copy + 'static,
    C: Deref<Target = Vec<T>> + Send + Sync + Sized + 'static,
>(
    _ctx: Context,
    contents: &C,
) -> Result<Buffer> {
    upload_array_buffer_impl(contents, None)
}

#[snoozy]
pub async fn upload_array_tex_buffer<
    T: Sized + Copy + 'static,
    C: Deref<Target = Vec<T>> + Send + Sync + Sized + 'static,
>(
    _ctx: Context,
    contents: &C,
    texture_format: &vk::Format,
) -> Result<Buffer> {
    upload_array_buffer_impl(contents, Some(*texture_format))
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
