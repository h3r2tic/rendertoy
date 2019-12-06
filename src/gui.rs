use crate::texture::{Texture, TextureKey};
use ash::version::DeviceV1_0;
use ash::vk;

pub fn create_imgui_render_pass(device: &ash::Device) -> vk::RenderPass {
    let renderpass_attachments = [vk::AttachmentDescription {
        format: vk::Format::R8G8B8A8_UNORM,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ..Default::default()
    }];
    let color_attachment_refs = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let dependencies = [vk::SubpassDependency {
        src_subpass: vk::SUBPASS_EXTERNAL,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ..Default::default()
    }];

    let subpasses = [vk::SubpassDescription::builder()
        .color_attachments(&color_attachment_refs)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .build()];

    let renderpass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&renderpass_attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    let renderpass = unsafe {
        device
            .create_render_pass(&renderpass_create_info, None)
            .unwrap()
    };

    renderpass
}

pub fn create_imgui_framebuffer(
    device: &ash::Device,
    render_pass: vk::RenderPass,
) -> (vk::Framebuffer, Texture) {
    let surface_resolution = unsafe { crate::vulkan::vk_all() }.surface_resolution;
    let tex = crate::backend::texture::create_texture(TextureKey::new(
        surface_resolution.width,
        surface_resolution.height,
        vk::Format::R8G8B8A8_UNORM,
    ));

    let framebuffer_attachments = [tex.rt_view];
    let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
        .render_pass(render_pass)
        .attachments(&framebuffer_attachments)
        .width(surface_resolution.width)
        .height(surface_resolution.height)
        .layers(1);

    let fb = unsafe { device.create_framebuffer(&frame_buffer_create_info, None) }
        .expect("create_framebuffer");

    (fb, tex)
}
