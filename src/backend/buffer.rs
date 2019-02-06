use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Eq, PartialEq, Hash, Clone, Copy, Serialize, Debug)]
pub struct BufferKey {
    pub size_bytes: usize,
}

#[derive(Clone)]
pub struct Buffer {
    pub buffer_id: u32,
    pub key: BufferKey,
    _allocation: std::sync::Arc<BufferAllocation>,
}

pub fn create_buffer(key: BufferKey) -> Buffer {
    let alloc = BufferAllocation::new(key);
    Buffer {
        buffer_id: alloc.buffer_id,
        key,
        _allocation: std::sync::Arc::new(alloc),
    }
}

#[derive(Clone)]
struct BufferAllocation {
    key: BufferKey,
    buffer_id: u32,
}

lazy_static! {
    static ref TRANSIENT_BUFFER_CACHE: Mutex<HashMap<BufferKey, Vec<BufferAllocation>>> =
        { Mutex::new(HashMap::new()) };
}

impl Drop for BufferAllocation {
    fn drop(&mut self) {
        TRANSIENT_BUFFER_CACHE
            .lock()
            .unwrap()
            .entry(self.key)
            .or_default()
            .push(self.clone());
    }
}

impl BufferAllocation {
    fn new(key: BufferKey) -> BufferAllocation {
        let mut texcache_lock = TRANSIENT_BUFFER_CACHE.lock().unwrap();
        let existing = texcache_lock.entry(key).or_default();

        if existing.is_empty() {
            //println!("Allocating new buffer: {:?}", key);

            unsafe {
                let mut buffer_id = 0;
                gl::GenBuffers(1, &mut buffer_id);
                gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, buffer_id);
                //gl::BufferData(gl::SHADER_STORAGE_BUFFER, key.size_bytes, std::ptr::null_mut(), gl::STATIC_COPY);

                gl::BufferStorage(
                    gl::SHADER_STORAGE_BUFFER,
                    key.size_bytes as isize,
                    std::ptr::null(),
                    gl::DYNAMIC_STORAGE_BIT,
                );

                BufferAllocation { key, buffer_id }
            }
        } else {
            existing.pop().unwrap()
        }
    }
}
