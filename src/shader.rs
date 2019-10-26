use crate::backend::{self, render_buffer::*};
use crate::blob::*;
use crate::buffer::Buffer;
use crate::gpu_debugger;
use crate::gpu_profiler;
use crate::texture::{Texture, TextureKey};
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
    pub fn new<T: Into<ShaderUniformValue> + 'static>(name: &str, value: T) -> ShaderUniformHolder {
        let mut s = DefaultSnoozyHash::default();
        whatever_hash(&value, &mut s);
        let shallow_hash = std::hash::Hasher::finish(&mut s);

        let value = value.into();

        ShaderUniformHolder {
            name: name.to_string(),
            shallow_hash,
            value,
        }
    }
}

pub type ShaderUniformBundle = Vec<ShaderUniformHolder>;

#[macro_export]
macro_rules! shader_uniforms {
    (@parse_name $name:ident) => {
        stringify!($name)
    };
    (@parse_name) => {
        ""
    };
    ($($($name:ident)? : $value:expr),* $(,)*) => {
        vec![
            $(ShaderUniformHolder::new(shader_uniforms!(@parse_name $($name)?), $value),)*
        ];
    }
}

pub struct ComputeShader {
    pub name: String,
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
    let name = std::path::Path::new(&path.asset_name)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or("unknown".to_string());

    Ok(ComputeShader {
        handle: backend::shader::make_program(&[shader_handle])?,
        name,
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
        .map(|a| ctx.get(&*a).map(|s| s.handle))
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

pub enum PlumberEvent<'a> {
    SetUniform(&'a ShaderUniformHolder),
    EnterScope,
    LeaveScope,
}

pub trait ShaderUniformPlumberCallback {
    fn plumb(&mut self, ctx: &mut Context, uniform: &ShaderUniformHolder) -> Result<()>;
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

    fn plumb_uniform(
        &mut self,
        ctx: &mut Context,
        program_handle: u32,
        uniform: &ShaderUniformHolder,
    ) -> Result<()> {
        let c_name = std::ffi::CString::new(uniform.name.clone()).unwrap();
        let loc = unsafe { gl::GetUniformLocation(program_handle, c_name.as_ptr()) };

        match uniform.value {
            ShaderUniformValue::Bundle(_) => {}
            ShaderUniformValue::BundleAsset(_) => {}

            ShaderUniformValue::TextureAsset(ref tex_asset) => {
                let tex = ctx.get(tex_asset)?;

                let size_loc = unsafe {
                    let size_name = uniform.name.clone() + "_size";
                    let c_name = std::ffi::CString::new(size_name).unwrap();
                    gl::GetUniformLocation(program_handle, c_name.as_ptr())
                };

                if size_loc != -1 {
                    unsafe {
                        gl::Uniform4f(
                            size_loc,
                            tex.key.width as f32,
                            tex.key.height as f32,
                            1.0 / tex.key.width as f32,
                            1.0 / tex.key.height as f32,
                        );
                    }
                }

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

                        //dbg!((&uniform.name, tex.texture_id));

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
                        } else {
                            panic!("unspupported sampler type: {:x}", type_gl);
                        }
                    }
                }
            }
            ShaderUniformValue::BufferAsset(ref buf_asset) => {
                let buf = ctx.get(buf_asset)?;

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

                        if gl::SAMPLER_BUFFER == type_gl
                            || gl::UNSIGNED_INT_SAMPLER_BUFFER == type_gl
                            || gl::INT_SAMPLER_BUFFER == type_gl
                        {
                            gl::ActiveTexture(gl::TEXTURE0 + self.img_unit as u32);
                            gl::BindTexture(
                                gl::TEXTURE_BUFFER,
                                buf.texture_id
                                    .expect("buffer doesn't have a texture buffer"),
                            );
                            gl::BindSampler(self.img_unit as u32, 0);
                            gl::Uniform1i(loc, self.img_unit);
                            self.img_unit += 1;
                        } else {
                            panic!(
                                "Buffer textures can only be bound to gsamplerBuffer; got {:x}",
                                type_gl
                            );
                        }
                    } else {
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

        Ok(())
    }

    fn plumb<UniformHandlerFn>(
        &mut self,
        ctx: &mut Context,
        program_handle: u32,
        uniforms: &Vec<ShaderUniformHolder>,
        uniform_handler_fn: &mut UniformHandlerFn,
    ) -> Result<()>
    where
        UniformHandlerFn:
            FnMut(&mut Context, &mut dyn ShaderUniformPlumberCallback, PlumberEvent) -> Result<()>,
    {
        impl ShaderUniformPlumberCallback for (&mut ShaderUniformPlumber, u32) {
            fn plumb(&mut self, ctx: &mut Context, uniform: &ShaderUniformHolder) -> Result<()> {
                self.0.plumb_uniform(ctx, self.1, uniform)
            }
        }

        macro_rules! scope_event {
            ($event_type: expr) => {
                let _ = uniform_handler_fn(ctx, &mut (&mut *self, program_handle), $event_type)?;
            };
        }

        // Do non-bundle values first so that the become visible to bundles
        for uniform in uniforms.iter() {
            match uniform.value {
                ShaderUniformValue::Bundle(_) => {}
                ShaderUniformValue::BundleAsset(_) => {}
                _ => {
                    let _ = uniform_handler_fn(
                        ctx,
                        &mut (&mut *self, program_handle),
                        PlumberEvent::SetUniform(uniform),
                    )?;
                }
            }
        }

        // Now process bundles
        for uniform in uniforms.iter() {
            match uniform.value {
                ShaderUniformValue::Bundle(ref bundle) => {
                    scope_event!(PlumberEvent::EnterScope);
                    self.plumb(ctx, program_handle, bundle, uniform_handler_fn)?;
                    scope_event!(PlumberEvent::LeaveScope);
                }
                ShaderUniformValue::BundleAsset(ref bundle) => {
                    let bundle = &*ctx.get(bundle)?;
                    scope_event!(PlumberEvent::EnterScope);
                    self.plumb(ctx, program_handle, bundle, uniform_handler_fn)?;
                    scope_event!(PlumberEvent::LeaveScope);
                }
                _ => {}
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
    let output_tex = backend::texture::create_texture(*key);

    let mut uniform_plumber = ShaderUniformPlumber::default();

    let cs = ctx.get(cs)?;

    let mut img_unit = {
        uniform_plumber.prepare(ctx, uniforms)?;

        unsafe {
            gl::UseProgram(cs.handle);
        }

        uniform_plumber.plumb(ctx, cs.handle, uniforms, &mut |ctx, plumber, event| {
            if let PlumberEvent::SetUniform(uniform) = event {
                plumber.plumb(ctx, uniform)
            } else {
                Ok(())
            }
        })?;
        uniform_plumber.img_unit
    };

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

        gpu_profiler::profile(&cs.name, || {
            gl::DispatchCompute(
                (dispatch_size.0 + work_group_size[0] as u32 - 1) / work_group_size[0] as u32,
                (dispatch_size.1 + work_group_size[1] as u32 - 1) / work_group_size[1] as u32,
                1,
            )
        });

        for i in 0..img_unit {
            gl::ActiveTexture(gl::TEXTURE0 + i as u32);
            gl::BindTexture(gl::TEXTURE_2D, 0);
        }
    }

    //dbg!(&cs.name);
    gpu_debugger::report_texture(&cs.name, output_tex.texture_id);
    //dbg!(output_tex.texture_id);

    Ok(output_tex)
}

#[snoozy]
pub fn raster_tex(
    ctx: &mut Context,
    key: &TextureKey,
    raster_pipe: &SnoozyRef<RasterPipeline>,
    uniforms: &Vec<ShaderUniformHolder>,
) -> Result<Texture> {
    let output_tex = backend::texture::create_texture(*key);
    let depth_buffer = create_render_buffer(RenderBufferKey {
        width: key.width,
        height: key.height,
        format: gl::DEPTH_COMPONENT32F,
    });

    let mut uniform_plumber = ShaderUniformPlumber::default();
    uniform_plumber.prepare(ctx, uniforms)?;

    let raster_pipe = ctx.get(raster_pipe)?;
    let mut img_unit = 0;

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
        gl::UseProgram(raster_pipe.handle);
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

        uniform_plumber.img_unit = img_unit;

        #[derive(Default)]
        struct MeshDrawData {
            index_buffer: Option<u32>,
            index_count: Option<u32>,
        }

        let mut mesh_stack = vec![MeshDrawData::default()];

        uniform_plumber.plumb(
            ctx,
            raster_pipe.handle,
            uniforms,
            &mut |ctx, plumber, event| match event {
                PlumberEvent::SetUniform(uniform) => {
                    match uniform.value {
                        ShaderUniformValue::BufferAsset(ref buf_asset)
                            if uniform.name == "mesh_index_buf" =>
                        {
                            let buf = ctx.get(buf_asset)?;
                            mesh_stack.last_mut().unwrap().index_buffer = Some(buf.buffer_id);
                        }
                        ShaderUniformValue::Uint32(ref value)
                            if uniform.name == "mesh_index_count" =>
                        {
                            mesh_stack.last_mut().unwrap().index_count = Some(*value);
                        }
                        _ => {}
                    }

                    plumber.plumb(ctx, uniform)
                }
                PlumberEvent::EnterScope => {
                    mesh_stack.push(Default::default());
                    Ok(())
                }
                PlumberEvent::LeaveScope => {
                    let mesh = mesh_stack.pop().unwrap();
                    if let Some(index_count) = mesh.index_count {
                        if let Some(index_buffer) = mesh.index_buffer {
                            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, index_buffer);
                            gl::DrawElements(
                                gl::TRIANGLES,
                                index_count as i32,
                                gl::UNSIGNED_INT,
                                std::ptr::null(),
                            );
                            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, 0);
                        } else {
                            gl::DrawArrays(gl::TRIANGLES, 0, index_count as i32);
                        }
                    }
                    Ok(())
                }
            },
        )?;

        gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
        gl::DeleteFramebuffers(1, &fb_handle);

        for i in 0..img_unit {
            gl::ActiveTexture(gl::TEXTURE0 + i as u32);
            gl::BindTexture(gl::TEXTURE_2D, 0);
        }
    }

    Ok(output_tex)
}
