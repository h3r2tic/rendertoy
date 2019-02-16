use super::blob::*;
use super::buffer::Buffer;
use super::texture::{Texture, TextureKey};
use crate::backend::{self, render_buffer::*};
use relative_path::{RelativePath, RelativePathBuf};
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
        Uint32(u32),
        Int32(i32),
        Ivec2((i32, i32)),
        Float32Asset(SnoozyRef<f32>),
        Uint32Asset(SnoozyRef<u32>),
        UsizeAsset(SnoozyRef<usize>),
        TextureAsset(SnoozyRef<Texture>),
        BufferAsset(SnoozyRef<Buffer>),
        Bundle(Vec<ShaderUniformHolder>),
        BundleAsset(SnoozyRef<Vec<ShaderUniformHolder>>),
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

pub type ShaderUniformBundle = Vec<ShaderUniformHolder>;

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
    type IncludeContext = AssetPath;

    fn get_include(
        &mut self,
        path: &str,
        include_context: &Self::IncludeContext,
    ) -> Result<(String, Self::IncludeContext)> {
        let asset_path: AssetPath = if let Some(crate_end) = path.find("::") {
            let crate_name = path.chars().take(crate_end).collect();
            let asset_name = path.chars().skip(crate_end + 2).collect();

            AssetPath {
                crate_name,
                asset_name,
            }
        } else {
            if let Some('/') = path.chars().next() {
                AssetPath {
                    crate_name: include_context.crate_name.clone(),
                    asset_name: path.chars().skip(1).collect(),
                }
            } else {
                let mut folder: RelativePathBuf = include_context.asset_name.clone().into();
                folder.pop();
                AssetPath {
                    crate_name: include_context.crate_name.clone(),
                    asset_name: folder.join(path).as_str().to_string(),
                }
            }
        };

        RelativePath::new(path);
        let blob = self.ctx.get(&load_blob(asset_path.clone()))?;
        String::from_utf8(blob.contents.clone())
            .map_err(|e| format_err!("{}", e))
            .map(|ok| (ok, asset_path))
    }
}

#[snoozy]
pub fn load_cs(ctx: &mut Context, path: &AssetPath) -> Result<ComputeShader> {
    let source = shader_prepper::process_file(
        &path.asset_name,
        &mut ShaderIncludeProvider { ctx: ctx },
        AssetPath {
            crate_name: path.crate_name.clone(),
            asset_name: String::new(),
        },
    )?;

    let shader_handle = backend::shader::make_shader(gl::COMPUTE_SHADER, &source)?;

    Ok(ComputeShader {
        handle: backend::shader::make_program(&[shader_handle])?,
    })
}

pub struct RasterSubShader {
    handle: u32,
}

impl Drop for RasterSubShader {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteShader(self.handle);
        }
    }
}

#[snoozy]
pub fn load_vs(ctx: &mut Context, path: &AssetPath) -> Result<RasterSubShader> {
    let source = shader_prepper::process_file(
        &path.asset_name,
        &mut ShaderIncludeProvider { ctx: ctx },
        AssetPath {
            crate_name: path.crate_name.clone(),
            asset_name: String::new(),
        },
    )?;

    Ok(RasterSubShader {
        handle: backend::shader::make_shader(gl::VERTEX_SHADER, &source)?,
    })
}

#[snoozy]
pub fn load_ps(ctx: &mut Context, path: &AssetPath) -> Result<RasterSubShader> {
    let source = shader_prepper::process_file(
        &path.asset_name,
        &mut ShaderIncludeProvider { ctx: ctx },
        AssetPath {
            crate_name: path.crate_name.clone(),
            asset_name: String::new(),
        },
    )?;

    Ok(RasterSubShader {
        handle: backend::shader::make_shader(gl::FRAGMENT_SHADER, &source)?,
    })
}

pub struct RasterPipeline {
    handle: u32,
}

#[snoozy]
pub fn make_raster_pipeline(
    ctx: &mut Context,
    shaders: &Vec<SnoozyRef<RasterSubShader>>,
) -> Result<RasterPipeline> {
    let shaders: Result<Vec<u32>> = shaders
        .iter()
        .map(|a| ctx.get(*a).map(|s| s.handle))
        .collect();

    Ok(RasterPipeline {
        handle: backend::shader::make_program(shaders?.as_slice())?,
    })
}

#[derive(Default)]
struct ShaderUniformPlumber {
    img_unit: i32,
    ssbo_unit: u32,
    index_count: Option<u32>,
}

impl ShaderUniformPlumber {
    fn prepare(&self, ctx: &mut Context, uniforms: &Vec<ShaderUniformHolder>) -> Result<()> {
        // Eval input parameters
        for uniform in uniforms.iter() {
            match uniform.value {
                ShaderUniformValue::TextureAsset(ref asset) => {
                    ctx.get(asset)?;
                }
                ShaderUniformValue::BufferAsset(ref asset) => {
                    ctx.get(asset)?;
                }
                ShaderUniformValue::Float32Asset(ref asset) => {
                    ctx.get(asset)?;
                }
                ShaderUniformValue::Uint32Asset(ref asset) => {
                    ctx.get(asset)?;
                }
                ShaderUniformValue::UsizeAsset(ref asset) => {
                    ctx.get(asset)?;
                }
                ShaderUniformValue::Bundle(ref bundle) => {
                    self.prepare(ctx, &bundle)?;
                }
                ShaderUniformValue::BundleAsset(ref bundle) => {
                    let bundle = &*ctx.get(bundle)?;
                    self.prepare(ctx, bundle)?;
                }
                ShaderUniformValue::Float32(_)
                | ShaderUniformValue::Uint32(_)
                | ShaderUniformValue::Int32(_)
                | ShaderUniformValue::Ivec2(_) => {}
            }
        }

        Ok(())
    }

    fn plumb(
        &mut self,
        ctx: &mut Context,
        program_handle: u32,
        uniforms: &Vec<ShaderUniformHolder>,
    ) -> Result<()> {
        for uniform in uniforms.iter() {
            let c_name = std::ffi::CString::new(uniform.name.clone()).unwrap();
            let loc = unsafe { gl::GetUniformLocation(program_handle, c_name.as_ptr()) };

            match uniform.value {
                ShaderUniformValue::TextureAsset(ref tex_asset) => {
                    let tex = ctx.get(tex_asset)?;

                    unsafe {
                        if loc != -1 {
                            let mut type_gl = 0;
                            let mut size = 0;
                            gl::GetActiveUniform(
                                program_handle,
                                loc as u32,
                                0,
                                std::ptr::null_mut(),
                                &mut size,
                                &mut type_gl,
                                std::ptr::null_mut(),
                            );

                            if gl::IMAGE_2D == type_gl {
                                let level = 0;
                                let layered = gl::FALSE;
                                gl::BindImageTexture(
                                    self.img_unit as u32,
                                    tex.texture_id,
                                    level,
                                    layered,
                                    0,
                                    gl::READ_ONLY,
                                    tex.key.format,
                                );
                                gl::Uniform1i(loc, self.img_unit);
                                self.img_unit += 1;
                            } else if gl::SAMPLER_2D == type_gl {
                                gl::ActiveTexture(gl::TEXTURE0 + self.img_unit as u32);
                                gl::BindTexture(gl::TEXTURE_2D, tex.texture_id);
                                gl::BindSampler(self.img_unit as u32, tex.sampler_id);
                                gl::Uniform1i(loc, self.img_unit);
                                self.img_unit += 1;
                            }
                        }
                    }
                }
                ShaderUniformValue::BufferAsset(ref buf_asset) => {
                    let buf = ctx.get(buf_asset)?;

                    unsafe {
                        let block_index = gl::GetProgramResourceIndex(
                            program_handle,
                            gl::SHADER_STORAGE_BLOCK,
                            c_name.as_ptr(),
                        );
                        if block_index != std::u32::MAX {
                            gl::ShaderStorageBlockBinding(
                                program_handle,
                                block_index,
                                self.ssbo_unit,
                            );
                            gl::BindBufferBase(
                                gl::SHADER_STORAGE_BUFFER,
                                self.ssbo_unit,
                                buf.buffer_id,
                            );
                            self.ssbo_unit += 1;
                        }
                    }
                }
                ShaderUniformValue::Bundle(ref bundle) => {
                    self.plumb(ctx, program_handle, bundle)?;
                }
                ShaderUniformValue::BundleAsset(ref bundle) => {
                    let bundle = &*ctx.get(bundle)?;
                    self.plumb(ctx, program_handle, bundle)?;
                }
                ShaderUniformValue::Float32(ref value) => unsafe {
                    gl::Uniform1f(loc, *value);
                },
                ShaderUniformValue::Int32(ref value) => unsafe {
                    gl::Uniform1i(loc, *value);
                },
                ShaderUniformValue::Uint32(ref value) => unsafe {
                    gl::Uniform1ui(loc, *value);

                    if uniform.name == "mesh_index_count" {
                        self.index_count = Some(*value);
                    }
                },
                ShaderUniformValue::Ivec2(ref value) => unsafe {
                    gl::Uniform2i(loc, value.0, value.1);
                },
                ShaderUniformValue::Float32Asset(ref asset) => unsafe {
                    gl::Uniform1f(loc, *ctx.get(asset)?);
                },
                ShaderUniformValue::Uint32Asset(ref asset) => unsafe {
                    gl::Uniform1ui(loc, *ctx.get(asset)?);
                },
                ShaderUniformValue::UsizeAsset(ref asset) => unsafe {
                    gl::Uniform1i(loc, *ctx.get(asset)? as i32);
                },
            }
        }

        Ok(())
    }
}

#[snoozy]
pub fn compute_tex(
    ctx: &mut Context,
    key: &TextureKey,
    cs: &SnoozyRef<ComputeShader>,
    uniforms: &Vec<ShaderUniformHolder>,
) -> Result<Texture> {
    let mut uniform_plumber = ShaderUniformPlumber::default();

    let cs = ctx.get(cs)?;

    let mut img_unit = {
        uniform_plumber.prepare(ctx, uniforms)?;

        unsafe {
            gl::UseProgram(cs.handle);
        }

        uniform_plumber.plumb(ctx, cs.handle, uniforms)?;
        uniform_plumber.img_unit
    };

    let output_tex = backend::texture::create_texture(*key);

    let dispatch_size = (key.width, key.height);

    unsafe {
        let level = 0;
        let layered = gl::FALSE;
        gl::BindImageTexture(
            img_unit as u32,
            output_tex.texture_id,
            level,
            layered,
            0,
            gl::WRITE_ONLY,
            key.format,
        );
        gl::Uniform1i(
            gl::GetUniformLocation(cs.handle, "outputTex\0".as_ptr() as *const i8),
            img_unit,
        );
        gl::Uniform4f(
            gl::GetUniformLocation(cs.handle, "outputTex_size\0".as_ptr() as *const i8),
            dispatch_size.0 as f32,
            dispatch_size.1 as f32,
            1f32 / dispatch_size.0 as f32,
            1f32 / dispatch_size.1 as f32,
        );
        img_unit += 1;

        let mut work_group_size: [i32; 3] = [0, 0, 0];
        gl::GetProgramiv(
            cs.handle,
            gl::COMPUTE_WORK_GROUP_SIZE,
            &mut work_group_size[0],
        );

        gl::DispatchCompute(
            (dispatch_size.0 + work_group_size[0] as u32 - 1) / work_group_size[0] as u32,
            (dispatch_size.1 + work_group_size[1] as u32 - 1) / work_group_size[1] as u32,
            1,
        );

        for i in 0..img_unit {
            gl::ActiveTexture(gl::TEXTURE0 + i as u32);
            gl::BindTexture(gl::TEXTURE_2D, 0);
        }
    }

    Ok(output_tex)
}

#[snoozy]
pub fn raster_tex(
    ctx: &mut Context,
    key: &TextureKey,
    raster_pipe: &SnoozyRef<RasterPipeline>,
    uniforms: &Vec<ShaderUniformHolder>,
) -> Result<Texture> {
    let mut uniform_plumber = ShaderUniformPlumber::default();

    let raster_pipe = ctx.get(raster_pipe)?;

    let mut img_unit = {
        uniform_plumber.prepare(ctx, uniforms)?;

        unsafe {
            gl::UseProgram(raster_pipe.handle);
        }

        uniform_plumber.plumb(ctx, raster_pipe.handle, uniforms)?;
        uniform_plumber.img_unit
    };

    let index_count = match uniform_plumber.index_count {
        Some(val) => val,
        None => {
            return Err(format_err!(
                "mesh_index_count: u32 not found in shader uniforms"
            ));
        }
    };

    let output_tex = backend::texture::create_texture(*key);
    let depth_buffer = create_render_buffer(RenderBufferKey {
        width: key.width,
        height: key.height,
        format: gl::DEPTH_COMPONENT24,
    });

    let fb_handle = {
        let mut handle: u32 = 0;
        unsafe {
            gl::GenFramebuffers(1, &mut handle);
            gl::BindFramebuffer(gl::FRAMEBUFFER, handle);

            gl::FramebufferTexture2D(
                gl::FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                output_tex.texture_id,
                0,
            );

            gl::FramebufferRenderbuffer(
                gl::FRAMEBUFFER,
                gl::DEPTH_ATTACHMENT,
                gl::RENDERBUFFER,
                depth_buffer.render_buffer_id,
            );

            gl::BindFramebuffer(gl::FRAMEBUFFER, handle);
        }
        handle
    };

    unsafe {
        gl::Uniform4f(
            gl::GetUniformLocation(raster_pipe.handle, "outputTex_size\0".as_ptr() as *const i8),
            key.width as f32,
            key.height as f32,
            1.0 / key.width as f32,
            1.0 / key.height as f32,
        );
        img_unit += 1;

        gl::Viewport(0, 0, key.width as i32, key.height as i32);
        gl::DepthFunc(gl::GEQUAL);
        gl::Enable(gl::DEPTH_TEST);
        gl::Disable(gl::CULL_FACE);

        gl::ClearColor(0.0, 0.0, 0.0, 0.0);
        gl::ClearDepth(0.0);
        gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

        gl::DrawArrays(gl::TRIANGLES, 0, index_count as i32);

        gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
        gl::DeleteFramebuffers(1, &fb_handle);

        for i in 0..img_unit {
            gl::ActiveTexture(gl::TEXTURE0 + i as u32);
            gl::BindTexture(gl::TEXTURE_2D, 0);
        }
    }

    Ok(output_tex)
}
