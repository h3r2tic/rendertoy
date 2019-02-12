use crate::{na, CameraMatrices, Matrix4, Vector2};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct ViewportConstants {
    pub view_to_clip: Matrix4,
    pub clip_to_view: Matrix4,
    pub world_to_view: Matrix4,
    pub view_to_world: Matrix4,
}

impl ViewportConstants {
    pub fn build(
        camera_matrices: CameraMatrices,
        width: u32,
        height: u32,
    ) -> VieportConstantBuilder {
        VieportConstantBuilder {
            width,
            height,
            camera_matrices,
            pixel_offset: Vector2::zeros(),
        }
    }
}

pub struct VieportConstantBuilder {
    width: u32,
    height: u32,
    camera_matrices: CameraMatrices,
    pixel_offset: Vector2,
}

impl VieportConstantBuilder {
    pub fn pixel_offset(mut self, v: Vector2) -> Self {
        self.pixel_offset = v;
        self
    }

    pub fn finish(self) -> ViewportConstants {
        let mut view_to_clip = self.camera_matrices.view_to_clip;
        view_to_clip.m13 = (2.0 * self.pixel_offset.x) / self.width as f32;
        view_to_clip.m23 = (2.0 * self.pixel_offset.y) / self.height as f32;

        let clip_to_view = na::convert::<_, na::Matrix4<f32>>(
            na::convert::<_, na::Matrix4<f64>>(view_to_clip)
                .try_inverse()
                .unwrap(),
        );

        ViewportConstants {
            view_to_clip,
            clip_to_view,
            world_to_view: self.camera_matrices.world_to_view,
            view_to_world: self.camera_matrices.view_to_world,
        }
    }
}
