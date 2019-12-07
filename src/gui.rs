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

// Based on https://github.com/ocornut/imgui/issues/707#issuecomment-430613104
pub fn setup_imgui_style(ctx: &mut imgui::Context) {
    let hi = |v: f32| [0.502, 0.075, 0.256, v];
    let med = |v: f32| [0.455, 0.198, 0.301, v];
    let low = |v: f32| [0.232, 0.201, 0.271, v];
    let bg = |v: f32| [0.200, 0.220, 0.270, v];
    let text = |v: f32| [0.860, 0.930, 0.890, v];

    let style = ctx.style_mut();
    style.colors[imgui::StyleColor::Text as usize] = text(0.78);
    style.colors[imgui::StyleColor::TextDisabled as usize] = text(0.28);
    style.colors[imgui::StyleColor::WindowBg as usize] = [0.13, 0.14, 0.17, 1.00];
    style.colors[imgui::StyleColor::ChildBg as usize] = bg(0.58);
    style.colors[imgui::StyleColor::PopupBg as usize] = bg(0.9);
    style.colors[imgui::StyleColor::Border as usize] = [0.31, 0.31, 1.00, 0.00];
    style.colors[imgui::StyleColor::BorderShadow as usize] = [0.00, 0.00, 0.00, 0.00];
    style.colors[imgui::StyleColor::FrameBg as usize] = bg(1.00);
    style.colors[imgui::StyleColor::FrameBgHovered as usize] = med(0.78);
    style.colors[imgui::StyleColor::FrameBgActive as usize] = med(1.00);
    style.colors[imgui::StyleColor::TitleBg as usize] = low(1.00);
    style.colors[imgui::StyleColor::TitleBgActive as usize] = hi(1.00);
    style.colors[imgui::StyleColor::TitleBgCollapsed as usize] = bg(0.75);
    style.colors[imgui::StyleColor::MenuBarBg as usize] = bg(0.47);
    style.colors[imgui::StyleColor::ScrollbarBg as usize] = bg(1.00);
    style.colors[imgui::StyleColor::ScrollbarGrab as usize] = [0.09, 0.15, 0.16, 1.00];
    style.colors[imgui::StyleColor::ScrollbarGrabHovered as usize] = med(0.78);
    style.colors[imgui::StyleColor::ScrollbarGrabActive as usize] = med(1.00);
    style.colors[imgui::StyleColor::CheckMark as usize] = [0.71, 0.22, 0.27, 1.00];
    style.colors[imgui::StyleColor::SliderGrab as usize] = [0.47, 0.77, 0.83, 0.14];
    style.colors[imgui::StyleColor::SliderGrabActive as usize] = [0.71, 0.22, 0.27, 1.00];
    style.colors[imgui::StyleColor::Button as usize] = [0.47, 0.77, 0.83, 0.14];
    style.colors[imgui::StyleColor::ButtonHovered as usize] = med(0.86);
    style.colors[imgui::StyleColor::ButtonActive as usize] = med(1.00);
    style.colors[imgui::StyleColor::Header as usize] = med(0.76);
    style.colors[imgui::StyleColor::HeaderHovered as usize] = med(0.86);
    style.colors[imgui::StyleColor::HeaderActive as usize] = hi(1.00);
    //style.colors[imgui::StyleColor::Column as usize] = [0.14, 0.16, 0.19, 1.00];
    //style.colors[imgui::StyleColor::ColumnHovered as usize] = med(0.78);
    //style.colors[imgui::StyleColor::ColumnActive as usize] = med(1.00);
    style.colors[imgui::StyleColor::ResizeGrip as usize] = [0.47, 0.77, 0.83, 0.04];
    style.colors[imgui::StyleColor::ResizeGripHovered as usize] = med(0.78);
    style.colors[imgui::StyleColor::ResizeGripActive as usize] = med(1.00);
    style.colors[imgui::StyleColor::PlotLines as usize] = text(0.63);
    style.colors[imgui::StyleColor::PlotLinesHovered as usize] = med(1.00);
    style.colors[imgui::StyleColor::PlotHistogram as usize] = text(0.63);
    style.colors[imgui::StyleColor::PlotHistogramHovered as usize] = med(1.00);
    style.colors[imgui::StyleColor::TextSelectedBg as usize] = med(0.43);
    style.colors[imgui::StyleColor::ModalWindowDimBg as usize] = bg(0.73);

    style.window_padding = [6.0, 4.0];
    style.window_rounding = 0.0;
    style.frame_padding = [5.0, 2.0];
    style.frame_rounding = 3.0;
    style.item_spacing = [7.0, 1.0];
    style.item_inner_spacing = [1.0, 1.0];
    style.touch_extra_padding = [0.0, 0.0];
    style.indent_spacing = 6.0;
    style.scrollbar_size = 12.0;
    style.scrollbar_rounding = 16.0;
    style.grab_min_size = 20.0;
    style.grab_rounding = 2.0;

    style.window_title_align[0] = 0.50;

    style.colors[imgui::StyleColor::Border as usize] = [0.539, 0.479, 0.255, 0.162];
    style.frame_border_size = 0.0;
    style.window_border_size = 1.0;
}
