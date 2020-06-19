use super::transient_resource::*;
use crate::{vk, vulkan::*};
use ash::version::DeviceV1_0;

#[derive(Eq, PartialEq, Hash, Clone, Copy, Serialize, Debug)]
pub struct BufferKey {
    pub size_bytes: usize,
    pub texture_format: Option<i32>,
}

impl BufferKey {
    pub fn new(size_bytes: usize, texture_format: Option<vk::Format>) -> Self {
        Self {
            size_bytes,
            texture_format: texture_format.map(|fmt| fmt.as_raw()),
        }
    }
}

#[derive(Clone)]
pub struct BufferAllocation {
    view: vk::BufferView,
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    allocation_info: vk_mem::AllocationInfo,
    bindless_index: u32,
}

#[derive(Clone)]
pub struct Buffer {
    pub view: vk::BufferView,
    pub buffer: vk::Buffer,
    pub key: BufferKey,
    pub bindless_index: u32,
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
            view: allocation.payload.view,
            buffer: allocation.payload.buffer,
            key: desc,
            bindless_index: allocation.payload.bindless_index,
            _allocation: allocation,
        }
    }

    fn allocate_payload(key: BufferKey) -> BufferAllocation {
        unsafe {
            let usage: vk::BufferUsageFlags = vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER;

            let mem_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::GpuOnly,
                ..Default::default()
            };

            let (buffer, allocation, allocation_info) = {
                let buffer_info = vk::BufferCreateInfo::builder()
                    .size(key.size_bytes as u64)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build();

                vk().allocator
                    .create_buffer(&buffer_info, &mem_info)
                    .expect("vma::create_buffer")
            };

            let (vk, vk_state) = vk_all();

            let view_format =
                vk::Format::from_raw(key.texture_format.unwrap_or(vk::Format::R8_UNORM.as_raw()));

            let view_info = vk::BufferViewCreateInfo::builder()
                .buffer(buffer)
                .format(view_format)
                .range(key.size_bytes as u64);
            let view = vk
                .device
                .create_buffer_view(&view_info.build(), None)
                .expect("create_buffer_view");

            let bindless_index = vk_state.register_buffer_bindless_index(view);

            BufferAllocation {
                view,
                buffer,
                allocation,
                allocation_info,
                bindless_index,
            }
        }
    }
}

pub fn create_buffer(key: BufferKey) -> Buffer {
    create_transient(key)
}
