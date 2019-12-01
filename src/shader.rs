use crate::backend::{self, render_buffer::*};
use crate::blob::*;
use crate::buffer::Buffer;
use crate::gpu_debugger;
use crate::gpu_profiler;
use crate::texture::{Texture, TextureKey};
use crate::vulkan::*;
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0, InstanceV1_1};
use ash::{vk, Device};
use relative_path::{RelativePath, RelativePathBuf};
use shader_prepper;
use snoozy::futures::future::{try_join_all, BoxFuture, FutureExt};
use snoozy::*;
use std::collections::HashMap;
use std::ffi::CStr;

macro_rules! def_shader_uniform_types {
    (@resolved_type SnoozyRef<ShaderUniformBundle>) => {
        ResolvedShaderUniformBundle
    };
    (@resolved_type SnoozyRef<$t:ty>) => {
        $t
    };
    (@resolved_type ShaderUniformBundle) => {
        ResolvedShaderUniformBundle
    };
    (@resolved_type $t:ty) => {
        $t
    };
    (@resolve $ctx:ident SnoozyRef<ShaderUniformBundle>, $v:ident) => {
        resolve($ctx.clone(), (*$ctx.get($v).await?).clone()).await?
    };
    (@resolve $ctx:ident SnoozyRef<$t:ty>, $v:ident) => {
        (*$ctx.get($v).await?).clone()
    };
    (@resolve $ctx:ident ShaderUniformBundle, $v:ident) => {
        resolve($ctx.clone(), $v.clone()).await?
    };
    (@resolve $ctx:ident $t:ty, $v:ident) => {
        (*$v).clone()
    };
    ($($name:ident($($type:tt)*)),* $(,)*) => {
		#[derive(Clone, Debug, Serialize)]
		pub enum ShaderUniformValue {
			$($name($($type)*)),*
		}

		pub enum ResolvedShaderUniformValue {
			$($name(def_shader_uniform_types!(@resolved_type $($type)*))),*
		}

        impl ShaderUniformValue {
            pub fn resolve<'a>(&'a self, ctx: Context) -> BoxFuture<'a, Result<ResolvedShaderUniformValue>> {
                async move {
                    match self {
                        $(ShaderUniformValue::$name(v) => Ok(ResolvedShaderUniformValue::$name(
                            def_shader_uniform_types!(@resolve ctx $($type)*, v)
                        ))),*
                    }
                }.boxed()
            }
		}

        $(
			impl From<$($type)*> for ShaderUniformValue {
				fn from(v: $($type)*) -> ShaderUniformValue {
					ShaderUniformValue::$name(v)
				}
			}
		)*
	}
}

def_shader_uniform_types! {
    Float32(f32),
    Uint32(u32),
    Int32(i32),
    Ivec2((i32, i32)),
    Vec4((f32, f32, f32, f32)),
    Bundle(ShaderUniformBundle),
    Float32Asset(SnoozyRef<f32>),
    Uint32Asset(SnoozyRef<u32>),
    UsizeAsset(SnoozyRef<usize>),
    TextureAsset(SnoozyRef<Texture>),
    BufferAsset(SnoozyRef<Buffer>),
    BundleAsset(SnoozyRef<ShaderUniformBundle>),
}

#[derive(Debug, Serialize, Clone)]
pub struct ShaderUniformHolder {
    name: String,
    value: ShaderUniformValue,
    shallow_hash: u64,
}

use std::hash::{Hash, Hasher};
impl Hash for ShaderUniformHolder {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.shallow_hash.hash(state);
    }
}

pub struct ResolvedShaderUniformHolder {
    name: String,
    value: ResolvedShaderUniformValue,
}

impl ShaderUniformHolder {
    pub fn new<T: Into<ShaderUniformValue> + 'static>(name: &str, value: T) -> ShaderUniformHolder {
        Self::from_name_value(name, value.into())
    }

    pub fn from_name_value(name: &str, value: ShaderUniformValue) -> ShaderUniformHolder {
        let mut s = DefaultSnoozyHash::default();
        whatever_hash(&value, &mut s);
        let shallow_hash = std::hash::Hasher::finish(&mut s);

        ShaderUniformHolder {
            name: name.to_string(),
            value,
            shallow_hash,
        }
    }

    pub async fn resolve(&self, ctx: Context) -> Result<ResolvedShaderUniformHolder> {
        Ok(ResolvedShaderUniformHolder {
            name: self.name.clone(),
            value: self.value.resolve(ctx.clone()).await?,
        })
    }
}

pub type ShaderUniformBundle = Vec<ShaderUniformHolder>;
pub type ResolvedShaderUniformBundle = Vec<ResolvedShaderUniformHolder>;

async fn resolve(
    ctx: Context,
    uniforms: Vec<ShaderUniformHolder>,
) -> Result<Vec<ResolvedShaderUniformHolder>> {
    // TODO: don't clone all the things.
    //
    try_join_all(uniforms.into_iter().map(|u| {
        let ctx = ctx.clone();
        tokio::task::spawn(async move { u.resolve(ctx).await })
    }))
    .await
    .expect("tokio join error")
    .into_iter()
    .collect()
}

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
        ]
    }
}

#[macro_export]
macro_rules! shader_uniform_bundle {
    (@parse_name $name:ident) => {
        stringify!($name)
    };
    (@parse_name) => {
        ""
    };
    ($($($name:ident)? : $value:expr),* $(,)*) => {
        ShaderUniformHolder::new("",
        vec![
            $(ShaderUniformHolder::new(shader_uniforms!(@parse_name $($name)?), $value),)*
        ])
    }
}

pub struct ComputeShader {
    pub name: String,
    pipeline: ComputePipeline,
    spirv_reflection: spirv_reflect::ShaderModule,
    reflection: ShaderReflection,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    local_size: (u32, u32, u32),
}

unsafe impl Send for ComputeShader {}
unsafe impl Sync for ComputeShader {}

#[derive(Debug)]
pub struct ShaderUniformReflection {
    pub location: i32,
    pub gl_type: u32,
}

#[derive(Debug)]
pub struct ShaderReflection {
    uniforms: HashMap<String, ShaderUniformReflection>,
}

fn reflect_shader(gfx: &crate::Gfx, program_handle: u32) -> ShaderReflection {
    unimplemented!()
    /*let mut uniform_count = 0i32;
    unsafe {
        gl.GetProgramiv(program_handle, gl::ACTIVE_UNIFORMS, &mut uniform_count);
    }

    let uniforms: HashMap<String, ShaderUniformReflection> = (0..uniform_count)
        .map(|index| unsafe {
            let mut name_len = 0;
            let mut gl_type = 0;
            let mut gl_size = 0;

            let mut name_str: Vec<u8> = vec![b'\0'; 128];
            gl.GetActiveUniform(
                program_handle,
                index as u32,
                127,
                &mut name_len,
                &mut gl_size,
                &mut gl_type,
                name_str.as_mut_ptr() as *mut i8,
            );

            let location = gl.GetUniformLocation(program_handle, name_str.as_ptr() as *const i8);

            let name = CStr::from_ptr(name_str.as_ptr() as *const i8)
                .to_string_lossy()
                .into_owned();
            let refl = ShaderUniformReflection { location, gl_type };

            (name, refl)
        })
        .collect();

    ShaderReflection { uniforms }*/
}

impl Drop for ComputeShader {
    fn drop(&mut self) {
        // TODO: defer
        /*unsafe {
            gl.DeleteProgram(self.handle);
        }*/
    }
}

struct ShaderIncludeProvider {
    ctx: Context,
}

impl<'a> shader_prepper::IncludeProvider for ShaderIncludeProvider {
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
        let blob =
            snoozy::futures::executor::block_on(self.ctx.get(&load_blob(asset_path.clone())))?;
        String::from_utf8(blob.contents.clone())
            .map_err(|e| format_err!("{}", e))
            .map(|ok| (ok, asset_path))
    }
}

fn get_shader_text(source: &[shader_prepper::SourceChunk]) -> String {
    let preamble =
        "#version 430\n#extension GL_EXT_samplerless_texture_functions : require\n".to_string();

    let mod_sources = source.iter().enumerate().map(|(i, s)| {
        let s = format!("#line 0 {}\n", i + 1) + &s.source;
        s
    });
    let mod_sources = std::iter::once(preamble).chain(mod_sources);
    let mod_sources: Vec<_> = mod_sources.collect();

    mod_sources.join("")
}

fn shaderc_compile_glsl(
    shader_name: &str,
    source: &[shader_prepper::SourceChunk],
) -> Result<shaderc::CompilationArtifact> {
    use shaderc;
    let source = get_shader_text(source);
    shaderc_compile_glsl_str(shader_name, &source)
}

fn shaderc_compile_glsl_str(
    shader_name: &str,
    source: &str,
) -> Result<shaderc::CompilationArtifact> {
    use shaderc;

    let mut compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.add_macro_definition("EP", Some("main"));
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    options.set_generate_debug_info();
    options.set_auto_bind_uniforms(true);
    let binary_result = compiler
        .compile_into_spirv(
            source,
            shaderc::ShaderKind::Compute,
            "shader.glsl",
            "main",
            Some(&options),
        )
        //.expect(&format!("{}::compile_into_spirv", shader_name));
        ?;

    assert_eq!(Some(&0x07230203), binary_result.as_binary().first());

    Ok(binary_result)
}

pub struct ComputePipeline {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

fn convert_spirv_reflect_err<T>(res: std::result::Result<T, &'static str>) -> Result<T> {
    match res {
        Ok(res) => Ok(res),
        Err(e) => bail!("SPIR-V reflection error: {}", e),
    }
}

fn reflect_spirv_shader(shader_code: &[u32]) -> Result<spirv_reflect::ShaderModule> {
    convert_spirv_reflect_err(spirv_reflect::ShaderModule::load_u32_data(shader_code))
}

fn generate_descriptor_set_layouts(
    refl: &spirv_reflect::ShaderModule,
) -> std::result::Result<Vec<vk::DescriptorSetLayout>, &'static str> {
    let mut result = Vec::new();

    let entry = Some("main");
    for descriptor_set in refl.enumerate_descriptor_sets(entry)?.iter() {
        let mut binding_flags: Vec<vk::DescriptorBindingFlagsEXT> = Vec::new();
        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = Vec::new();

        for binding in descriptor_set.bindings.iter() {
            use spirv_reflect::types::descriptor::ReflectDescriptorType;

            let desc_type = match binding.descriptor_type {
                ReflectDescriptorType::UniformBuffer => {
                    Some(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                }
                ReflectDescriptorType::StorageImage => Some(vk::DescriptorType::STORAGE_IMAGE),
                ReflectDescriptorType::SampledImage => Some(vk::DescriptorType::SAMPLED_IMAGE),
                ReflectDescriptorType::StorageBuffer => Some(vk::DescriptorType::STORAGE_BUFFER),
                ReflectDescriptorType::UniformTexelBuffer => {
                    Some(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                }
                _ => {
                    print!("\tunsupported");
                    None
                }
            };

            if let Some(desc_type) = desc_type {
                let sub_type = *binding.type_description.as_ref().unwrap().op;
                //println!("{:#?}", binding);

                let mut binding_builder = vk::DescriptorSetLayoutBinding::builder()
                    .descriptor_count(binding.count)
                    .descriptor_type(desc_type)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .binding(binding.binding);

                // TODO
                if desc_type == vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                    && sub_type == spirv_headers::Op::TypeRuntimeArray
                {
                    binding_flags.push(
                        vk::DescriptorBindingFlagsEXT::VARIABLE_DESCRIPTOR_COUNT
                            | vk::DescriptorBindingFlagsEXT::PARTIALLY_BOUND
                            | vk::DescriptorBindingFlagsEXT::UPDATE_UNUSED_WHILE_PENDING,
                    );
                    binding_builder = binding_builder.descriptor_count(1 << 18);
                } else {
                    binding_flags.push(vk::DescriptorBindingFlagsEXT::empty());
                }

                bindings.push(binding_builder.build());
            }
        }

        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
            .binding_flags(&binding_flags)
            .build();

        let descriptor_set_layout = unsafe {
            vk_device()
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder()
                        .bindings(&bindings)
                        .push_next(&mut binding_flags)
                        .build(),
                    None,
                )
                .unwrap()
        };

        result.push(descriptor_set_layout);
    }

    Ok(result)
}

fn create_compute_pipeline(
    vk_device: &Device,
    descriptor_set_layouts: &[vk::DescriptorSetLayout],
    shader_code: &[u32],
) -> Result<ComputePipeline> {
    use std::ffi::{CStr, CString};
    use std::io::Cursor;

    let shader_entry_name = CString::new("main").unwrap();

    let layout_create_info =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);

    unsafe {
        let shader_module = vk_device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::builder().code(&shader_code),
                None,
            )
            .unwrap();

        let stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(shader_module)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(&shader_entry_name);

        let pipeline_layout = vk_device
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_create_info.build())
            .layout(pipeline_layout);

        let t0 = std::time::Instant::now();
        // TODO: pipeline cache
        let pipeline = vk_device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None)
            .expect("pipeline")[0];
        println!("create_compute_pipelines took {:?}", t0.elapsed());

        Ok(ComputePipeline {
            pipeline_layout,
            pipeline,
        })
    }
}

fn get_cs_local_size_from_spirv(spirv: &[u32]) -> Result<(u32, u32, u32)> {
    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_words(spirv, &mut loader).unwrap();
    let module = loader.module();

    for inst in module.global_inst_iter() {
        if spirv_headers::Op::ExecutionMode == inst.class.opcode {
            let local_size = &inst.operands[2..5];
            use rspirv::dr::Operand::LiteralInt32;

            if let &[LiteralInt32(x), LiteralInt32(y), LiteralInt32(z)] = local_size {
                return Ok((x, y, z));
            } else {
                bail!("Could not parse the ExecutionMode SPIR-V op");
            }
        }
    }

    bail!("Could not find a ExecutionMode SPIR-V op");
}

// Pack descriptor sets so that they use small consecutive integers, e.g. sets [0, 5, 31] become [0, 1, 2]
fn compact_descriptor_sets(refl: &mut spirv_reflect::ShaderModule) {
    let entry = Some("main");
    let sets = refl
        .enumerate_descriptor_sets(entry)
        .expect("enumerate_descriptor_sets");
    let mut set_order: Vec<_> = sets.iter().enumerate().map(|(i, s)| (i, s.set)).collect();
    set_order.sort_by_key(|(i, set_idx)| *set_idx);
    for (new_idx, (old_idx, _)) in set_order.into_iter().enumerate() {
        refl.change_descriptor_set_number(&sets[old_idx], new_idx as u32);
    }
}

fn load_cs_impl(name: String, source: &[shader_prepper::SourceChunk]) -> Result<ComputeShader> {
    let t0 = std::time::Instant::now();

    let refl = {
        let spirv = shaderc_compile_glsl(&name, source)?;
        println!("shaderc_compile_glsl took {:?}", t0.elapsed());

        let mut refl = reflect_spirv_shader(spirv.as_binary())?;
        compact_descriptor_sets(&mut refl);
        refl
    };

    let spirv_binary = refl.get_code();

    let local_size = get_cs_local_size_from_spirv(&spirv_binary)?;

    let descriptor_set_layouts = convert_spirv_reflect_err(generate_descriptor_set_layouts(&refl))?;
    let pipeline = create_compute_pipeline(vk_device(), &descriptor_set_layouts, &spirv_binary)?;

    let reflection = ShaderReflection {
        uniforms: Default::default(),
    };

    Ok(ComputeShader {
        name,
        pipeline,
        reflection,
        spirv_reflection: refl,
        descriptor_set_layouts,
        local_size,
    })
}

#[snoozy]
pub async fn load_cs(ctx: Context, path: &AssetPath) -> Result<ComputeShader> {
    let source = shader_prepper::process_file(
        &path.asset_name,
        &mut ShaderIncludeProvider { ctx: ctx.clone() },
        AssetPath {
            crate_name: path.crate_name.clone(),
            asset_name: String::new(),
        },
    )?;

    let name = std::path::Path::new(&path.asset_name)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or("unknown".to_string());

    load_cs_impl(name, &source)
}

#[snoozy]
pub async fn load_cs_from_string(
    _ctx: Context,
    source: &String,
    name: &String,
) -> Result<ComputeShader> {
    let source = [shader_prepper::SourceChunk {
        source: source.clone(),
        file: "no-file".to_owned(),
        line_offset: 0,
    }];

    let name = std::path::Path::new(&name)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or("unknown".to_string());

    load_cs_impl(name, &source)
}

pub struct RasterSubShader {
    handle: u32,
}

impl Drop for RasterSubShader {
    fn drop(&mut self) {
        // TODO: defer
        /*unsafe {
            gl.DeleteShader(self.handle);
        }*/
    }
}

#[snoozy]
pub async fn load_vs(ctx: Context, path: &AssetPath) -> Result<RasterSubShader> {
    let source = shader_prepper::process_file(
        &path.asset_name,
        &mut ShaderIncludeProvider { ctx: ctx.clone() },
        AssetPath {
            crate_name: path.crate_name.clone(),
            asset_name: String::new(),
        },
    )?;

    /*with_gl(|gl| {
        Ok(RasterSubShader {
            handle: backend::shader::make_shader(gfx, gl::VERTEX_SHADER, &source)?,
        })
    })*/
    unimplemented!()
}

#[snoozy]
pub async fn load_ps(ctx: Context, path: &AssetPath) -> Result<RasterSubShader> {
    let source = shader_prepper::process_file(
        &path.asset_name,
        &mut ShaderIncludeProvider { ctx: ctx.clone() },
        AssetPath {
            crate_name: path.crate_name.clone(),
            asset_name: String::new(),
        },
    )?;

    /*with_gl(|gl| {
        Ok(RasterSubShader {
            handle: backend::shader::make_shader(gfx, gl::FRAGMENT_SHADER, &source)?,
        })
    })*/
    unimplemented!()
}

pub struct RasterPipeline {
    handle: u32,
    reflection: ShaderReflection,
}

#[snoozy]
pub async fn make_raster_pipeline(
    ctx: Context,
    shaders_in: &Vec<SnoozyRef<RasterSubShader>>,
) -> Result<RasterPipeline> {
    let mut shaders = Vec::with_capacity(shaders_in.len());
    for a in shaders_in.iter() {
        shaders.push(ctx.get(&*a).await?.handle);
    }

    /*with_gl(|gl| {
        let handle = backend::shader::make_program(gfx, shaders.as_slice())?;
        let reflection = reflect_shader(gfx, handle);

        Ok(RasterPipeline { handle, reflection })
    })*/
    unimplemented!()
}

#[derive(Default)]
struct ShaderUniformPlumber {
    img_unit: i32,
    ssbo_unit: u32,
    ubo_unit: u32,
    index_count: Option<u32>,
    warnings: Vec<String>,
}

pub enum PlumberEvent {
    SetUniform {
        name: String,
        value: ResolvedShaderUniformValue,
    },
    EnterScope,
    LeaveScope,
}

impl ShaderUniformPlumber {
    /*fn plumb_uniform(
        &mut self,
        gfx: &crate::Gfx,
        program_handle: u32,
        reflection: &ShaderReflection,
        name: &str,
        value: &ResolvedShaderUniformValue,
    ) {
        unimplemented!()
        /*let c_name = std::ffi::CString::new(name.clone()).unwrap();

        macro_rules! get_uniform_no_warn {
            () => {
                reflection.uniforms.get(name)
            };
        }

        macro_rules! get_uniform {
            () => {{
                if let Some(u) = reflection.uniforms.get(name) {
                    Some(u)
                } else {
                    self.warnings
                        .push(format!("Shader uniform not found: {}", name).to_owned());
                    None
                }
            }};
        }

        match value {
            ResolvedShaderUniformValue::Bundle(_) => {}
            ResolvedShaderUniformValue::BundleAsset(_) => {}

            ResolvedShaderUniformValue::TextureAsset(ref tex) => {
                if let Some(loc) = reflection.uniforms.get(&(name.to_owned() + "_size")) {
                    unsafe {
                        gl.Uniform4f(
                            loc.location,
                            tex.key.width as f32,
                            tex.key.height as f32,
                            1.0 / tex.key.width as f32,
                            1.0 / tex.key.height as f32,
                        );
                    }
                }

                unsafe {
                    if let Some(loc) = get_uniform!() {
                        if gl::IMAGE_2D == loc.gl_type {
                            let level = 0;
                            let layered = gl::FALSE;
                            gl.BindImageTexture(
                                self.img_unit as u32,
                                tex.texture_id,
                                level,
                                layered,
                                0,
                                gl::READ_ONLY,
                                tex.key.format,
                            );
                            gl.Uniform1i(loc.location, self.img_unit);
                            self.img_unit += 1;
                        } else if gl::SAMPLER_2D == loc.gl_type {
                            gl.ActiveTexture(gl::TEXTURE0 + self.img_unit as u32);
                            gl.BindTexture(gl::TEXTURE_2D, tex.texture_id);
                            gl.BindSampler(self.img_unit as u32, tex.sampler_id);
                            gl.Uniform1i(loc.location, self.img_unit);
                            self.img_unit += 1;
                        } else {
                            panic!("unspupported sampler type: {:x}", loc.gl_type);
                        }
                    }
                }
            }
            ResolvedShaderUniformValue::BufferAsset(ref buf) => {
                let u_block_index =
                    unsafe { gl.GetUniformBlockIndex(program_handle, c_name.as_ptr()) };

                let ss_block_index = unsafe {
                    gl.GetProgramResourceIndex(
                        program_handle,
                        gl::SHADER_STORAGE_BLOCK,
                        c_name.as_ptr(),
                    )
                };

                if u_block_index != std::u32::MAX {
                    unsafe {
                        gl.UniformBlockBinding(program_handle, u_block_index, self.ubo_unit);
                        gl.BindBufferBase(gl::UNIFORM_BUFFER, self.ubo_unit, buf.buffer_id);
                    }
                    self.ubo_unit += 1;
                } else if ss_block_index != std::u32::MAX {
                    unsafe {
                        gl.ShaderStorageBlockBinding(
                            program_handle,
                            ss_block_index,
                            self.ssbo_unit,
                        );
                        gl.BindBufferBase(gl::SHADER_STORAGE_BUFFER, self.ssbo_unit, buf.buffer_id);
                    }
                    self.ssbo_unit += 1;
                } else {
                    unsafe {
                        if let Some(loc) = get_uniform_no_warn!() {
                            if gl::SAMPLER_BUFFER == loc.gl_type
                                || gl::UNSIGNED_INT_SAMPLER_BUFFER == loc.gl_type
                                || gl::INT_SAMPLER_BUFFER == loc.gl_type
                            {
                                gl.ActiveTexture(gl::TEXTURE0 + self.img_unit as u32);
                                gl.BindTexture(
                                    gl::TEXTURE_BUFFER,
                                    buf.texture_id
                                        .expect("buffer doesn't have a texture buffer"),
                                );
                                gl.BindSampler(self.img_unit as u32, 0);
                                gl.Uniform1i(loc.location, self.img_unit);
                                self.img_unit += 1;
                            } else {
                                panic!(
                                    "Buffer textures can only be bound to gsamplerBuffer; got {:x}",
                                    loc.gl_type
                                );
                            }
                        }
                    }
                }
            }
            ResolvedShaderUniformValue::Float32(value) => unsafe {
                if let Some(loc) = get_uniform!() {
                    gl.Uniform1f(loc.location, *value);
                }
            },
            ResolvedShaderUniformValue::Int32(value) => unsafe {
                if let Some(loc) = get_uniform!() {
                    gl.Uniform1i(loc.location, *value);
                }
            },
            ResolvedShaderUniformValue::Uint32(value) => unsafe {
                if name == "mesh_index_count" {
                    self.index_count = Some(*value);
                } else {
                    if let Some(loc) = get_uniform!() {
                        gl.Uniform1ui(loc.location, *value);
                    }
                }
            },
            ResolvedShaderUniformValue::Ivec2(value) => unsafe {
                if let Some(loc) = get_uniform!() {
                    gl.Uniform2i(loc.location, value.0, value.1);
                }
            },
            ResolvedShaderUniformValue::Float32Asset(value) => unsafe {
                if let Some(loc) = get_uniform!() {
                    gl.Uniform1f(loc.location, *value);
                }
            },
            ResolvedShaderUniformValue::Uint32Asset(value) => unsafe {
                if let Some(loc) = get_uniform!() {
                    gl.Uniform1ui(loc.location, *value);
                }
            },
            ResolvedShaderUniformValue::UsizeAsset(value) => unsafe {
                if let Some(loc) = get_uniform!() {
                    gl.Uniform1i(loc.location, *value as i32);
                }
            },
        }*/
    }*/
}

fn flatten_uniforms(
    mut uniforms: Vec<ResolvedShaderUniformHolder>,
    sink: &mut impl FnMut(PlumberEvent),
) {
    macro_rules! scope_event {
        ($event_type: expr) => {
            uniform_handler_fn($event_type);
        };
    }

    // Do non-bundle values first so that they become visible to bundle handlers
    for uniform in uniforms.iter_mut() {
        match uniform.value {
            ResolvedShaderUniformValue::Bundle(_) => {}
            ResolvedShaderUniformValue::BundleAsset(_) => {}
            ResolvedShaderUniformValue::TextureAsset(_) => {
                let name = std::mem::replace(&mut uniform.name, String::new());
                let value = if let ResolvedShaderUniformValue::TextureAsset(value) =
                    std::mem::replace(&mut uniform.value, ResolvedShaderUniformValue::Int32(0))
                {
                    value
                } else {
                    panic!()
                };

                let tex_size_uniform = (
                    value.key.width as f32,
                    value.key.height as f32,
                    1f32 / value.key.width as f32,
                    1f32 / value.key.height as f32,
                );

                sink(PlumberEvent::SetUniform {
                    name: name.clone() + "_size",
                    value: ResolvedShaderUniformValue::Vec4(tex_size_uniform),
                });

                sink(PlumberEvent::SetUniform {
                    name,
                    value: ResolvedShaderUniformValue::TextureAsset(value),
                });
            }
            _ => {
                let name = std::mem::replace(&mut uniform.name, String::new());
                let value =
                    std::mem::replace(&mut uniform.value, ResolvedShaderUniformValue::Int32(0));

                sink(PlumberEvent::SetUniform { name, value });
            }
        }
    }

    // Now process bundles
    for uniform in uniforms.into_iter() {
        match uniform.value {
            ResolvedShaderUniformValue::Bundle(bundle)
            | ResolvedShaderUniformValue::BundleAsset(bundle) => {
                sink(PlumberEvent::EnterScope);
                flatten_uniforms(bundle, sink);
                sink(PlumberEvent::LeaveScope);
            }
            _ => {}
        }
    }
}

fn update_descriptor_sets(
    device: &Device,
    refl: &spirv_reflect::ShaderModule,
    descriptor_sets: &[vk::DescriptorSet],
    uniforms: &HashMap<String, ResolvedShaderUniformValue>,
) -> std::result::Result<Vec<u32>, &'static str> {
    use std::cell::RefCell;

    let mut ds_writes = Vec::new();
    let mut ds_offsets = Vec::new();

    #[derive(Default)]
    pub struct Cache {
        ds_image_info: RefCell<Vec<[vk::DescriptorImageInfo; 1]>>,
        ds_buffer_info: RefCell<Vec<[vk::DescriptorBufferInfo; 1]>>,
        ds_buffer_views: RefCell<Vec<[vk::BufferView; 1]>>,
    }

    // TODO: cache those in thread-locals
    thread_local! {
        pub static CACHE: Cache = Default::default();
    }

    CACHE.with(|cache| {
        let mut ds_image_info = cache.ds_image_info.borrow_mut();
        let mut ds_buffer_info = cache.ds_buffer_info.borrow_mut();
        let mut ds_buffer_views = cache.ds_buffer_views.borrow_mut();

        ds_image_info.clear();
        ds_image_info.reserve(uniforms.len());

        ds_buffer_info.clear();
        ds_buffer_info.reserve(uniforms.len());

        ds_buffer_views.clear();
        ds_buffer_views.reserve(uniforms.len());

        let entry = Some("main");
        for (ds_idx, descriptor_set) in refl.enumerate_descriptor_sets(entry)?.iter().enumerate() {
            let ds = descriptor_sets[0];
            for binding in descriptor_set.bindings.iter() {
                use spirv_reflect::types::descriptor::ReflectDescriptorType;

                match binding.descriptor_type {
                    ReflectDescriptorType::UniformBuffer => {
                        let buffer_bytes = binding.block.size as usize;
                        let (buffer_handle, buffer_offset, buffer_contents) = unsafe { vk_frame() }
                            .uniforms
                            .allocate(buffer_bytes)
                            .expect("failed to allocate uniform buffer");

                        for member in binding.block.members.iter() {
                            if let Some(value) = uniforms.get(&member.name) {
                                let dst_mem = &mut buffer_contents[member.absolute_offset as usize
                                    ..(member.absolute_offset + member.size) as usize];

                                match value {
                                    ResolvedShaderUniformValue::Float32(value)
                                    | ResolvedShaderUniformValue::Float32Asset(value) => {
                                        dst_mem.copy_from_slice(&(*value).to_ne_bytes());
                                    }
                                    ResolvedShaderUniformValue::Uint32(value)
                                    | ResolvedShaderUniformValue::Uint32Asset(value) => {
                                        dst_mem.copy_from_slice(&(*value).to_ne_bytes());
                                    }
                                    ResolvedShaderUniformValue::Int32(value) => {
                                        dst_mem.copy_from_slice(&(*value).to_ne_bytes());
                                    }
                                    ResolvedShaderUniformValue::Ivec2(value) => {
                                        dst_mem.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(
                                                std::mem::transmute(&value.0 as *const i32),
                                                2 * 4,
                                            )
                                        });
                                    }
                                    ResolvedShaderUniformValue::Vec4(value) => {
                                        dst_mem.copy_from_slice(unsafe {
                                            std::slice::from_raw_parts(
                                                std::mem::transmute(&value.0 as *const f32),
                                                4 * 4,
                                            )
                                        });
                                    }
                                    _ => {
                                        dbg!(member);
                                        unimplemented!();
                                    }
                                }
                            }
                        }

                        let buffer_info = [vk::DescriptorBufferInfo::builder()
                            .buffer(buffer_handle)
                            .range(buffer_bytes as u64)
                            .build()];
                        ds_buffer_info.push(buffer_info);
                        let buffer_info = ds_buffer_info.last().unwrap();

                        ds_offsets.push(buffer_offset as u32);
                        ds_writes.push(
                            vk::WriteDescriptorSet::builder()
                                .dst_set(ds)
                                .dst_binding(binding.binding)
                                .dst_array_element(0)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                                .buffer_info(buffer_info)
                                .build(),
                        );
                    }
                    ReflectDescriptorType::SampledImage => {
                        if let Some(ResolvedShaderUniformValue::TextureAsset(value)) =
                            uniforms.get(&binding.name)
                        {
                            let image_info = [vk::DescriptorImageInfo::builder()
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(value.view)
                                .build()];
                            ds_image_info.push(image_info);
                            let image_info = ds_image_info.last().unwrap();

                            ds_writes.push(
                                vk::WriteDescriptorSet::builder()
                                    .dst_set(ds)
                                    .dst_binding(binding.binding)
                                    .dst_array_element(0)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .image_info(image_info)
                                    .build(),
                            )
                        }
                    }
                    ReflectDescriptorType::StorageImage => {
                        if let Some(ResolvedShaderUniformValue::TextureAsset(value)) =
                            uniforms.get(&binding.name)
                        {
                            let image_info = [vk::DescriptorImageInfo::builder()
                                .image_layout(vk::ImageLayout::GENERAL)
                                .image_view(value.storage_view)
                                .build()];
                            ds_image_info.push(image_info);
                            let image_info = ds_image_info.last().unwrap();

                            ds_writes.push(
                                vk::WriteDescriptorSet::builder()
                                    .dst_set(ds)
                                    .dst_binding(binding.binding)
                                    .dst_array_element(0)
                                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                    .image_info(image_info)
                                    .build(),
                            )
                        }
                    }
                    ReflectDescriptorType::StorageBuffer => {
                        if let Some(ResolvedShaderUniformValue::BufferAsset(value)) =
                            uniforms.get(&binding.type_description.as_ref().unwrap().type_name)
                        {
                            let buffer_info = [vk::DescriptorBufferInfo::builder()
                                .buffer(value.buffer)
                                .range(vk::WHOLE_SIZE)
                                .build()];
                            ds_buffer_info.push(buffer_info);
                            let buffer_info = ds_buffer_info.last().unwrap();

                            ds_writes.push(
                                vk::WriteDescriptorSet::builder()
                                    .dst_set(ds)
                                    .dst_binding(binding.binding)
                                    .dst_array_element(0)
                                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                    .buffer_info(buffer_info)
                                    .build(),
                            );
                        }
                    }
                    ReflectDescriptorType::UniformTexelBuffer => {
                        if let Some(ResolvedShaderUniformValue::BufferAsset(value)) =
                            uniforms.get(&binding.name)
                        {
                            ds_buffer_views.push([value.view]);
                            let buffer_view = ds_buffer_views.last().unwrap();

                            ds_writes.push(
                                vk::WriteDescriptorSet::builder()
                                    .dst_set(ds)
                                    .dst_binding(binding.binding)
                                    .dst_array_element(0)
                                    .descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                                    .texel_buffer_view(buffer_view)
                                    .build(),
                            );
                        } else {
                            if "all_buffers" == binding.name {
                                // Nothing to do; updated elsewhere
                            } else {
                                // TODO
                                panic!("Could not find resource to bind {}", binding.name);
                            }
                        }
                    }
                    _ => print!("\tunsupported"),
                }
            }
        }

        if !ds_writes.is_empty() {
            unsafe { device.update_descriptor_sets(&ds_writes, &[]) };
        }

        Ok(ds_offsets)
    })
}

#[snoozy]
pub async fn compute_tex(
    ctx: Context,
    key: &TextureKey,
    cs: &SnoozyRef<ComputeShader>,
    uniforms: &Vec<ShaderUniformHolder>,
) -> Result<Texture> {
    let output_tex = backend::texture::create_texture(*key);

    let cs = ctx.get(cs).await?;
    let mut uniforms = resolve(ctx, uniforms.clone()).await?;

    uniforms.push(ResolvedShaderUniformHolder {
        name: "outputTex".to_owned(),
        value: ResolvedShaderUniformValue::TextureAsset(output_tex.clone()),
    });

    let device = vk_device();
    let vk_frame = unsafe { vk_frame() };

    let mut flattened_uniforms: HashMap<String, ResolvedShaderUniformValue> = HashMap::new();
    flatten_uniforms(uniforms, &mut |e| {
        if let PlumberEvent::SetUniform { name, value } = e {
            flattened_uniforms.insert(name, value);
        }
    });

    let (descriptor_sets, dynamic_offsets) = unsafe {
        let descriptor_sets = {
            let descriptor_pool = vk_frame.descriptor_pool.lock().unwrap();
            let sets = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&cs.descriptor_set_layouts)
                    .build(),
            )?;
            drop(descriptor_pool);
            sets
        };

        let dynamic_offsets = update_descriptor_sets(
            device,
            &cs.spirv_reflection,
            &descriptor_sets,
            &flattened_uniforms,
        )
        .unwrap();

        (descriptor_sets, dynamic_offsets)
    };

    let cb = vk_frame.command_buffer.lock().unwrap();
    let cb: vk::CommandBuffer = cb.cb;

    unsafe {
        vk_all().record_image_barrier(
            cb,
            ImageBarrier::new(
                output_tex.image,
                vk_sync::AccessType::Nothing,
                vk_sync::AccessType::ComputeShaderWrite,
            )
            .with_discard(true),
        );

        let mut descriptor_sets = descriptor_sets;

        // TODO; "all_buffers" hack.
        if descriptor_sets.len() == 2 {
            descriptor_sets[1] = vk_all().bindless_buffers_descriptor_set;
        }

        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, cs.pipeline.pipeline);
        device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            cs.pipeline.pipeline_layout,
            0,
            &descriptor_sets,
            &dynamic_offsets,
        );

        device.cmd_dispatch(
            cb,
            (key.width + cs.local_size.0 - 1) / cs.local_size.0,
            (key.height + cs.local_size.1 - 1) / cs.local_size.1,
            1,
        );

        vk_all().record_image_barrier(
            cb,
            ImageBarrier::new(
                output_tex.image,
                vk_sync::AccessType::ComputeShaderWrite,
                vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
            ),
        );
    }

    /*for warning in uniform_plumber.warnings.iter() {
        crate::rtoy_show_warning(format!("{}: {}", cs.name, warning));
    }*/

    /*unsafe {
        let level = 0;
        let layered = gl::FALSE;
        gl.BindImageTexture(
            img_unit as u32,
            output_tex.texture_id,
            level,
            layered,
            0,
            gl::WRITE_ONLY,
            key.format,
        );
        gl.Uniform1i(
            gl.GetUniformLocation(cs.handle, "outputTex\0".as_ptr() as *const i8),
            img_unit,
        );
        gl.Uniform4f(
            gl.GetUniformLocation(cs.handle, "outputTex_size\0".as_ptr() as *const i8),
            dispatch_size.0 as f32,
            dispatch_size.1 as f32,
            1f32 / dispatch_size.0 as f32,
            1f32 / dispatch_size.1 as f32,
        );
        img_unit += 1;

        let mut work_group_size: [i32; 3] = [0, 0, 0];
        gl.GetProgramiv(
            cs.handle,
            gl::COMPUTE_WORK_GROUP_SIZE,
            &mut work_group_size[0],
        );

        gpu_profiler::profile(gfx, &cs.name, || {
            gl.DispatchCompute(
                (dispatch_size.0 + work_group_size[0] as u32 - 1) / work_group_size[0] as u32,
                (dispatch_size.1 + work_group_size[1] as u32 - 1) / work_group_size[1] as u32,
                1,
            )
        });

        for i in 0..img_unit {
            gl.ActiveTexture(gl::TEXTURE0 + i as u32);
            gl.BindTexture(gl::TEXTURE_2D, 0);
        }
    }*/

    //dbg!(&cs.name);
    gpu_debugger::report_texture(&cs.name, output_tex.view);
    //dbg!(output_tex.texture_id);

    Ok(output_tex)
}

#[snoozy]
pub async fn raster_tex(
    ctx: Context,
    key: &TextureKey,
    raster_pipe: &SnoozyRef<RasterPipeline>,
    uniforms: &Vec<ShaderUniformHolder>,
) -> Result<Texture> {
    let uniforms = resolve(ctx.clone(), uniforms.clone()).await?;
    let raster_pipe = ctx.get(raster_pipe).await?;

    unimplemented!()
    /*with_gl(|gl| {
        let output_tex = backend::texture::create_texture(gfx, *key);
        let depth_buffer = create_render_buffer(
            gl,
            RenderBufferKey {
                width: key.width,
                height: key.height,
                format: gl::DEPTH_COMPONENT32F,
            },
        );

        let mut uniform_plumber = ShaderUniformPlumber::default();
        let mut img_unit = 0;

        let fb_handle = {
            let mut handle: u32 = 0;
            unsafe {
                gl.GenFramebuffers(1, &mut handle);
                gl.BindFramebuffer(gl::FRAMEBUFFER, handle);

                gl.FramebufferTexture2D(
                    gl::FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0,
                    gl::TEXTURE_2D,
                    output_tex.texture_id,
                    0,
                );

                gl.FramebufferRenderbuffer(
                    gl::FRAMEBUFFER,
                    gl::DEPTH_ATTACHMENT,
                    gl::RENDERBUFFER,
                    depth_buffer.render_buffer_id,
                );

                gl.BindFramebuffer(gl::FRAMEBUFFER, handle);
            }
            handle
        };

        unsafe {
            gl.UseProgram(raster_pipe.handle);
            gl.Uniform4f(
                gl.GetUniformLocation(raster_pipe.handle, "outputTex_size\0".as_ptr() as *const i8),
                key.width as f32,
                key.height as f32,
                1.0 / key.width as f32,
                1.0 / key.height as f32,
            );
            img_unit += 1;

            gl.Viewport(0, 0, key.width as i32, key.height as i32);
            gl.DepthFunc(gl::GEQUAL);
            gl.Enable(gl::DEPTH_TEST);
            gl.Disable(gl::CULL_FACE);

            gl.ClearColor(0.0, 0.0, 0.0, 0.0);
            gl.ClearDepth(0.0);
            gl.Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            uniform_plumber.img_unit = img_unit;

            #[derive(Default)]
            struct MeshDrawData {
                index_buffer: Option<u32>,
                index_count: Option<u32>,
            }

            let mut mesh_stack = vec![MeshDrawData::default()];

            uniform_plumber.plumb(
                gl,
                raster_pipe.handle,
                &raster_pipe.reflection,
                &uniforms,
                &mut |plumber, event| match event {
                    PlumberEvent::SetUniform { name, value } => {
                        match value {
                            ResolvedShaderUniformValue::BufferAsset(buf)
                                if name == "mesh_index_buf" =>
                            {
                                mesh_stack.last_mut().unwrap().index_buffer = Some(buf.buffer_id);
                            }
                            ResolvedShaderUniformValue::Uint32(value)
                                if name == "mesh_index_count" =>
                            {
                                mesh_stack.last_mut().unwrap().index_count = Some(*value);
                            }
                            _ => {}
                        }

                        plumber.plumb(gfx, name, value)
                    }
                    PlumberEvent::EnterScope => {
                        mesh_stack.push(Default::default());
                    }
                    PlumberEvent::LeaveScope => {
                        let mesh = mesh_stack.pop().unwrap();
                        if let Some(index_count) = mesh.index_count {
                            if let Some(index_buffer) = mesh.index_buffer {
                                gl.BindBuffer(gl::ELEMENT_ARRAY_BUFFER, index_buffer);
                                gl.DrawElements(
                                    gl::TRIANGLES,
                                    index_count as i32,
                                    gl::UNSIGNED_INT,
                                    std::ptr::null(),
                                );
                                gl.BindBuffer(gl::ELEMENT_ARRAY_BUFFER, 0);
                            } else {
                                gl.DrawArrays(gl::TRIANGLES, 0, index_count as i32);
                            }
                        }
                    }
                },
            );

            gl.BindFramebuffer(gl::FRAMEBUFFER, 0);
            gl.DeleteFramebuffers(1, &fb_handle);

            for i in 0..img_unit {
                gl.ActiveTexture(gl::TEXTURE0 + i as u32);
                gl.BindTexture(gl::TEXTURE_2D, 0);
            }
        }

        Ok(output_tex)
    })*/
}
