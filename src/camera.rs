use crate::{na, na::UnitQuaternion, FrameState, Matrix4, Point3, Vector3, VirtualKeyCode};

#[derive(PartialEq, Clone)]
pub struct CameraMatrices {
    pub view_to_clip: Matrix4,
    pub world_to_view: Matrix4,
    pub view_to_world: Matrix4,
}

pub trait Camera {
    type InputType;

    fn update<InputType: Into<Self::InputType>>(&mut self, input: InputType, time: f32);
    fn calc_matrices(&self) -> CameraMatrices;
}

impl<T: Camera> From<&T> for CameraMatrices {
    fn from(camera: &T) -> CameraMatrices {
        camera.calc_matrices()
    }
}

pub struct FirstPersonCamera {
    // Degrees
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub fov: f32,

    pub position: Point3,
    pub near_dist: f32,
    pub aspect: f32,

    pub interp_rot: UnitQuaternion<f32>,
    pub interp_pos: Point3,
}

pub struct FirstPersonCameraInput {
    move_vec: Vector3,
    yaw_delta: f32,
    pitch_delta: f32,
}

fn highp_invert_matrix(m: &Matrix4) -> Option<Matrix4> {
    let m: na::Matrix4<f64> = na::convert(*m);
    m.try_inverse().map(|m| na::convert::<_, na::Matrix4<f32>>(m))
}

impl<'a> From<&FrameState<'a>> for FirstPersonCameraInput {
    fn from(frame_state: &FrameState<'a>) -> FirstPersonCameraInput {
        let mut yaw_delta = 0.0;
        let mut pitch_delta = 0.0;

        if (frame_state.mouse.button_mask & 4) == 4 {
            yaw_delta = -0.1 * frame_state.mouse.delta.x;
            pitch_delta = -0.1 * frame_state.mouse.delta.y;
        }

        let move_speed = 12.0;
        let mut move_vec = Vector3::zeros();

        if frame_state.keys.is_down(VirtualKeyCode::W) {
            move_vec.z += -move_speed;
        }
        if frame_state.keys.is_down(VirtualKeyCode::S) {
            move_vec.z += move_speed;
        }
        if frame_state.keys.is_down(VirtualKeyCode::A) {
            move_vec.x += -move_speed;
        }
        if frame_state.keys.is_down(VirtualKeyCode::D) {
            move_vec.x += move_speed;
        }

        FirstPersonCameraInput {
            move_vec,
            yaw_delta,
            pitch_delta,
        }
    }
}

impl FirstPersonCamera {
    fn rotate_yaw(&mut self, delta: f32) {
        self.yaw = (self.yaw + delta) % 720_f32;
    }

    fn rotate_pitch(&mut self, delta: f32) {
        self.pitch += delta;
        if self.pitch < -90.0 {
            self.pitch = -90.0
        }
        if self.pitch > 90.0 {
            self.pitch = 90.0
        }
    }

    fn translate(&mut self, local_v: &Vector3) {
        let rotation = self.calc_rotation_quat();
        self.position += rotation * local_v;
    }

    fn calc_rotation_quat(&self) -> UnitQuaternion<f32> {
        let yaw_rot: UnitQuaternion<f32> =
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), self.yaw.to_radians());
        let pitch_rot: UnitQuaternion<f32> =
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), self.pitch.to_radians());
        let roll_rot: UnitQuaternion<f32> =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), self.roll.to_radians());
        yaw_rot * (pitch_rot * roll_rot)
    }

    pub fn new(position: Point3) -> FirstPersonCamera {
        FirstPersonCamera {
            yaw: 0_f32,
            pitch: 0_f32,
            roll: 0_f32,
            fov: 45_f32,
            position,
            near_dist: 0.1_f32,
            aspect: 1.6_f32,
            interp_rot: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -0.0f32.to_radians()),
            interp_pos: position,
        }
    }
}

impl Camera for FirstPersonCamera {
    type InputType = FirstPersonCameraInput;

    fn update<InputType: Into<Self::InputType>>(&mut self, input: InputType, time: f32) {
        let input = input.into();
        self.translate(&input.move_vec);
        self.rotate_pitch(input.pitch_delta);
        self.rotate_yaw(input.yaw_delta);

        let target_quat = self.calc_rotation_quat();
        let rot_interp = 1.0 - (-time * 30.0).exp();
        let pos_interp = 1.0 - (-time * 16.0).exp();
        //self.interp_rot = self.interp_rot.slerp(&target_quat, rot_interp);
        self.interp_rot = target_quat;//self.interp_rot.slerp(&target_quat, rot_interp);
        self.interp_rot.renormalize();
        self.interp_pos = self
            .interp_pos
            .coords
            .lerp(&self.position.coords, pos_interp)
            .into();
    }

    fn calc_matrices(&self) -> CameraMatrices {
        let view_to_clip = {
            let fov = self.fov.to_radians();
            let znear = self.near_dist;

            let h = (0.5 * fov).cos() / (0.5 * fov).sin();
            let w = h / self.aspect;

            let mut m = Matrix4::zeros();
            m.m11 = w;
            m.m22 = h;
            m.m34 = znear;
            m.m43 = -1.0;
            m
        };

        let rotation = self.interp_rot.to_homogeneous();

        let view_to_world = {
            let translation = Matrix4::new_translation(&Vector3::new(
                self.interp_pos.x,
                self.interp_pos.y,
                self.interp_pos.z,
            ));
            translation * rotation
        };

        let world_to_view = highp_invert_matrix(&view_to_world).unwrap();

        CameraMatrices {
            view_to_clip: view_to_clip,
            world_to_view: world_to_view,
            view_to_world: view_to_world,
        }
    }
}

pub struct CameraConvergenceEnforcer<CameraType: Camera> {
    camera: CameraType,
    prev_matrices: CameraMatrices,
    frozen_matrices: CameraMatrices,
    prev_error: f64,
    is_converged: bool,
}

impl<CameraType: Camera> CameraConvergenceEnforcer<CameraType> {
    pub fn new(camera: CameraType) -> Self {
        let matrices = camera.calc_matrices();
        Self {
            camera,
            prev_matrices: matrices.clone(),
            frozen_matrices: matrices,
            prev_error: -1.0,
            is_converged: true,
        }
    }

    pub fn is_converged(&self) -> bool {
        self.is_converged
    }
}

impl<CameraType: Camera> Camera for CameraConvergenceEnforcer<CameraType> {
    type InputType = CameraType::InputType;

    fn update<InputType: Into<Self::InputType>>(&mut self, input: InputType, time: f32) {
        self.camera.update(input, time);

        let new_matrices = self.camera.calc_matrices();
        let view_to_clip: na::Matrix4<f64> = na::convert(new_matrices.view_to_clip);

        let error = if let Some(clip_to_view) = view_to_clip.try_inverse() {
            let cs_corners: [na::Vector4<f64>; 4] = [
                na::Vector4::new(-1.0, -1.0, 1e-5, 1.0),
                na::Vector4::new(1.0, -1.0, 1e-5, 1.0),
                na::Vector4::new(-1.0, 1.0, 1e-5, 1.0),
                na::Vector4::new(1.0, 1.0, 1e-5, 1.0),
            ];

            let clip_to_prev_clip =
                na::convert::<_, na::Matrix4<f64>>(self.prev_matrices.view_to_clip)
                    * na::convert::<_, na::Matrix4<f64>>(self.prev_matrices.world_to_view)
                    * na::convert::<_, na::Matrix4<f64>>(new_matrices.view_to_world)
                    * clip_to_view;

            let error: f64 = cs_corners
                .into_iter()
                .map(|cs_cur| {
                    let cs_prev = clip_to_prev_clip * cs_cur;
                    let cs_prev = cs_prev * (1.0 / cs_prev.w);
                    (cs_prev.xy() - cs_cur.xy()).norm()
                })
                .sum();
            error
        } else {
            std::f64::MAX
        };

        if error > 0.001 || error > self.prev_error * 1.05 + 1e-5 {
            self.frozen_matrices = new_matrices.clone();
            self.is_converged = false;
        } else {
            self.is_converged = true;
        }

        self.prev_matrices = new_matrices;
        self.prev_error = error;
    }

    fn calc_matrices(&self) -> CameraMatrices {
        self.frozen_matrices.clone()
    }
}
