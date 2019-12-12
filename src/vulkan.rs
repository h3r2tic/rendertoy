pub use crate::vk_backend_state::*;
pub use crate::vk_render_device::*;
use ash::version::DeviceV1_0;
use ash::vk;
use std::sync::{Arc, RwLock};

mod vk_backend_internals {
    use super::*;
    static mut VK_RENDER_DEVICE: Option<VkRenderDevice> = None;
    static mut VK_BACKEND_STATE: Option<RwLock<Arc<VkBackendState>>> = None;

    pub fn initialize_vulkan_backend(
        window: &winit::Window,
        graphics_debugging: bool,
        vsync: bool,
    ) {
        unsafe {
            assert!(VK_RENDER_DEVICE.is_none());
            assert!(VK_BACKEND_STATE.is_none());
        }

        let device = VkRenderDevice::new(window, graphics_debugging, vsync)
            .expect("VkRenderDevice creation failed");
        let bs = VkBackendState::new(&device, window, graphics_debugging, vsync)
            .expect("VkBackendState creation failed");

        unsafe {
            VK_RENDER_DEVICE = Some(device);
            VK_BACKEND_STATE = Some(RwLock::new(Arc::new(bs)));
        }
    }

    pub fn vk() -> &'static VkRenderDevice {
        unsafe {
            VK_RENDER_DEVICE
                .as_ref()
                .expect("Vulkan backend not initialized yet!")
        }
    }

    pub fn vk_state() -> impl std::ops::Deref<Target = VkBackendState> {
        let arc: Arc<_> = unsafe { VK_BACKEND_STATE.as_ref() }
            .expect("Vulkan backend not initialized yet!")
            .try_read()
            .expect("Cannot get a lock of the vulkan backend. It is being used exclusively")
            .clone();
        arc
    }

    pub fn vk_all() -> (
        &'static VkRenderDevice,
        impl std::ops::Deref<Target = VkBackendState>,
    ) {
        (vk(), vk_state())
    }

    pub fn with_vk_state_mut<R, Cb: Fn(&mut VkBackendState) -> R>(cb: Cb) -> R {
        let mut arc: &mut Arc<_> = &mut *unsafe { VK_BACKEND_STATE.as_ref() }
            .expect("Vulkan backend not initialized yet!")
            .try_write()
            .expect("Cannot mutably acquire the vulkan backend. The lock is being held.");
        cb(&mut *Arc::get_mut(&mut arc).expect(
            "Cannot mutably acquire the vulkan backend. A reference has been illegally retained.",
        ))
    }

    pub fn vk_add_setup_command(f: impl FnOnce(&VkRenderDevice, &VkFrameData) + Send + 'static) {
        if unsafe { VK_BACKEND_STATE.is_none() } || vk_state().current_frame_data_idx.is_none() {
            // If we haven't started rendering yet, delay this.
            VK_SETUP_COMMANDS.lock().unwrap().push(Box::new(f));
        } else {
            // Otherwise do it now
            let (vk, vk_state) = vk_all();
            f(vk, vk_state.current_frame());
        }
    }
}

pub use vk_backend_internals::*;

pub fn vk_resize(width: u32, height: u32) -> bool {
    with_vk_state_mut(|vk_state| {
        let vk = vk();
        unsafe { vk.device.device_wait_idle() }.unwrap();

        let mut create_info = vk_state.swapchain_create_info;
        create_info.surface_resolution = vk::Extent2D { width, height };

        vk_state.frame_data = Vec::new();
        vk_state.swapchain = None;

        vk_state.swapchain = create_swapchain(
            &vk.device,
            vk.pdevice,
            &vk.swapchain_loader,
            &vk.surface_loader,
            vk.surface,
            create_info,
        );

        if vk_state.swapchain.is_some() {
            vk_state.swapchain_create_info = create_info;
            vk_state.create_frame_data(vk);
            true
        } else {
            false
        }
    })
}
