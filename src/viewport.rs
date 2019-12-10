use crate::{CameraMatrices, Matrix4, Vector2};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct ViewConstants {
    pub view_to_clip: Matrix4,
    pub clip_to_view: Matrix4,
    pub view_to_sample: Matrix4,
    pub sample_to_view: Matrix4,
    pub world_to_view: Matrix4,
    pub view_to_world: Matrix4,
    pub sample_offset_pixels: Vector2,
    pub sample_offset_clip: Vector2,
}

impl ViewConstants {
    pub fn build<CamMat: Into<CameraMatrices>>(
        camera_matrices: CamMat,
        width: u32,
        height: u32,
    ) -> VieportConstantBuilder {
        VieportConstantBuilder {
            width,
            height,
            camera_matrices: camera_matrices.into(),
            pixel_offset: Vector2::zeros(),
        }
    }

    pub fn set_pixel_offset(&mut self, v: Vector2, width: u32, height: u32) {
        let sample_offset_pixels = v;
        let sample_offset_clip =
            Vector2::new((2.0 * v.x) / width as f32, (2.0 * v.y) / height as f32);

        let mut jitter_matrix = Matrix4::identity();
        jitter_matrix.m14 = -sample_offset_clip.x;
        jitter_matrix.m24 = -sample_offset_clip.y;

        let mut jitter_matrix_inv = Matrix4::identity();
        jitter_matrix_inv.m14 = sample_offset_clip.x;
        jitter_matrix_inv.m24 = sample_offset_clip.y;

        let view_to_sample = jitter_matrix * self.view_to_clip;
        let sample_to_view = self.clip_to_view * jitter_matrix_inv;

        self.view_to_sample = view_to_sample;
        self.sample_to_view = sample_to_view;
        self.sample_offset_pixels = sample_offset_pixels;
        self.sample_offset_clip = sample_offset_clip;
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

    pub fn build(self) -> ViewConstants {
        let view_to_clip = self.camera_matrices.view_to_clip;
        let clip_to_view = self.camera_matrices.clip_to_view;

        let mut res = ViewConstants {
            view_to_clip,
            clip_to_view,
            view_to_sample: Matrix4::zeros(),
            sample_to_view: Matrix4::zeros(),
            world_to_view: self.camera_matrices.world_to_view,
            view_to_world: self.camera_matrices.view_to_world,
            sample_offset_pixels: Vector2::zeros(),
            sample_offset_clip: Vector2::zeros(),
        };

        res.set_pixel_offset(self.pixel_offset, self.width, self.height);
        res
    }
}
