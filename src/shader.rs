use super::blob::*;
use super::texture::{Texture, TextureKey};
use crate::backend;
use shader_prepper;
use snoozy::*;

macro_rules! def_shader_uniform_types {
    (pub enum $enum_name:ident { $($name:ident($type:ty)),* $(,)* }) => {
		#[derive(Debug, Serialize)]
		pub enum $enum_name {
			$($name($type)),*
		}

		$(
			impl From<$type> for $enum_name {
				fn from(v: $type) -> $enum_name {
					$enum_name::$name(v)
				}
			}
		)*
	}
}

def_shader_uniform_types! {
    pub enum ShaderUniformValue {
        Float32(f32),
        Int32(i32),
        Ivec2((i32, i32)),
        Float32Asset(SnoozyRef<f32>),
        UsizeAsset(SnoozyRef<usize>),
        TextureAsset(SnoozyRef<Texture>),
    }
}

#[derive(Debug, Serialize)]
pub struct ShaderUniformHolder {
    name: String,
    value: ShaderUniformValue,
    shallow_hash: u64,
}

impl ShaderUniformHolder {
    pub fn new<T: Into<ShaderUniformValue>>(name: &str, value: T) -> ShaderUniformHolder {
        let value = value.into();

        ShaderUniformHolder {
            name: name.to_string(),
            shallow_hash: calculate_serialized_hash(&value),
            value,
        }
    }
}

#[macro_export]
macro_rules! shader_uniforms {
    ($($name:tt : $value:expr),* $(,)*) => {
		vec![
			$(ShaderUniformHolder::new($name, $value),)*
		];
	}
}

pub struct ComputeShader {
    handle: u32,
}

impl Drop for ComputeShader {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteProgram(self.handle);
        }
    }
}

struct ShaderIncludeProvider<'a> {
    ctx: &'a mut Context,
}

impl<'a> shader_prepper::IncludeProvider for ShaderIncludeProvider<'a> {
    fn get_include(&mut self, path: &str) -> Result<String> {
        let blob = self.ctx.get(&load_file("shaders/".to_string() + path))?;
        String::from_utf8(blob.contents.clone()).map_err(|e| format_err!("{}", e))
    }
}

snoozy! {
    fn load_cs(ctx: &mut Context, path: &String) -> Result<ComputeShader> {
        let source = shader_prepper::process_file(&path, &mut ShaderIncludeProvider {
            ctx: ctx
        })?;

        let shader_handle = backend::shader::make_shader(gl::COMPUTE_SHADER, &source)?;

        Ok(ComputeShader { handle: backend::shader::make_program(&[shader_handle])? })
    }
}

snoozy! {
    fn compute_tex(ctx: &mut Context, key: &TextureKey, cs: &SnoozyRef<ComputeShader>, uniforms: &Vec<ShaderUniformHolder>) -> Result<Texture> {
        // Eval input parameters
        for uniform in uniforms.iter() {
            match uniform.value {
                ShaderUniformValue::TextureAsset(ref asset) => {
                    ctx.get(asset)?;
                },
                ShaderUniformValue::Float32Asset(ref asset) =>{
                    ctx.get(asset)?;
                },
                ShaderUniformValue::UsizeAsset(ref asset) =>{
                    ctx.get(asset)?;
                },
                _ => (),
            }
        }

        let cs = ctx.get(cs)?;

        unsafe {
            gl::UseProgram(cs.handle);
        }

        let output_tex = backend::texture::create_texture(*key);

        let mut img_unit : i32 = 0;

        for uniform in uniforms.iter() {
            let c_name = std::ffi::CString::new(uniform.name.clone()).unwrap();
            let loc = unsafe { gl::GetUniformLocation(cs.handle, c_name.as_ptr()) };

            match uniform.value {
                ShaderUniformValue::TextureAsset(ref tex_asset) => {
                    let tex = ctx.get(tex_asset)?;

                    unsafe {
                        if loc != -1 {
                            let mut type_gl = 0;
                            let mut size = 0;
                            gl::GetActiveUniform(cs.handle, loc as u32, 0, std::ptr::null_mut(), &mut size, &mut type_gl, std::ptr::null_mut());

                            if gl::IMAGE_2D == type_gl {
                                let level = 0;
                                let layered = gl::FALSE;
                                gl::BindImageTexture(img_unit as u32, tex.texture_id, level, layered, 0, gl::READ_ONLY, tex.key.format);
                                gl::Uniform1i(loc, img_unit);
                                img_unit += 1;
                            }
                            else if gl::SAMPLER_2D == type_gl {
                                gl::ActiveTexture(gl::TEXTURE0 + img_unit as u32);
                                gl::BindTexture(gl::TEXTURE_2D, tex.texture_id);
                                gl::BindSampler(img_unit as u32, tex.sampler_id);
                                gl::Uniform1i(loc, img_unit);
                                img_unit += 1;
                            }
                        }
                    }
                },
                ShaderUniformValue::Float32(ref value) => {
                    unsafe { gl::Uniform1f(loc, *value); }
                },
                ShaderUniformValue::Int32(ref value) => {
                    unsafe { gl::Uniform1i(loc, *value); }
                },
                ShaderUniformValue::Ivec2(ref value) => {
                    unsafe { gl::Uniform2i(loc, value.0, value.1); }
                },
                ShaderUniformValue::Float32Asset(ref asset) => {
                    unsafe { gl::Uniform1f(loc, *ctx.get(asset)?); }
                },
                ShaderUniformValue::UsizeAsset(ref asset) => {
                    unsafe { gl::Uniform1i(loc, *ctx.get(asset)? as i32); }
                },
            }
        }

        let dispatch_size = (key.width, key.height);

        unsafe {
            let level = 0;
            let layered = gl::FALSE;
            gl::BindImageTexture(img_unit as u32, output_tex.texture_id, level, layered, 0, gl::WRITE_ONLY, key.format);
            gl::Uniform1i(gl::GetUniformLocation(cs.handle, "outputTex\0".as_ptr() as *const i8), img_unit);
            gl::Uniform4f(
                gl::GetUniformLocation(cs.handle, "outputTex_size\0".as_ptr() as *const i8),
                dispatch_size.0 as f32, dispatch_size.1 as f32,
                1f32 / dispatch_size.0 as f32, 1f32 / dispatch_size.1 as f32
            );
            img_unit += 1;

            let mut work_group_size : [i32; 3] = [0, 0, 0];
            gl::GetProgramiv(cs.handle, gl::COMPUTE_WORK_GROUP_SIZE, &mut work_group_size[0]);

            gl::DispatchCompute(
                (dispatch_size.0 + work_group_size[0] as u32 - 1) / work_group_size[0] as u32,
                (dispatch_size.1 + work_group_size[1] as u32 - 1) / work_group_size[1] as u32,
                1);

            for i in 0..img_unit {
                gl::ActiveTexture(gl::TEXTURE0 + i as u32);
                gl::BindTexture(gl::TEXTURE_2D, 0);
            }
        }

        Ok(output_tex)
    }
}
