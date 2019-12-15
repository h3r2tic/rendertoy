use crate::blob::*;
use crate::buffer::Buffer;
use crate::gpu_debugger;
use crate::texture::{Texture, TextureKey};
use crate::vulkan::*;
use ash::version::DeviceV1_0;
use ash::{vk, Device};
use relative_path::{RelativePath, RelativePathBuf};
use shader_prepper;
use snoozy::futures::future::{try_join_all, BoxFuture, FutureExt};
use snoozy::*;
use std::collections::{HashMap, HashSet};

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

        $(
			impl From<$($type)*> for ShaderUniformValue {
				fn from(v: $($type)*) -> ShaderUniformValue {
					ShaderUniformValue::$name(v)
				}
			}
		)*
	}
}

pub enum ResolvedShaderUniformValue {
    Float32(f32),
    Uint32(u32),
    Int32(i32),
    Usize(usize),
    Ivec2((i32, i32)),
    Vec4((f32, f32, f32, f32)),
    Texture(Texture),
    Buffer(Buffer),
    Bundle(ResolvedShaderUniformBundle),
    RwTexture(Texture),
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

impl ShaderUniformValue {
    pub fn resolve<'a>(
        &'a self,
        mut ctx: Context,
    ) -> BoxFuture<'a, Result<ResolvedShaderUniformValue>> {
        async move {
            match self {
                ShaderUniformValue::Float32(v) => Ok(ResolvedShaderUniformValue::Float32(*v)),
                ShaderUniformValue::Uint32(v) => Ok(ResolvedShaderUniformValue::Uint32(*v)),
                ShaderUniformValue::Int32(v) => Ok(ResolvedShaderUniformValue::Int32(*v)),
                ShaderUniformValue::Ivec2(v) => Ok(ResolvedShaderUniformValue::Ivec2(*v)),
                ShaderUniformValue::Vec4(v) => Ok(ResolvedShaderUniformValue::Vec4(*v)),
                ShaderUniformValue::Bundle(v) => Ok(ResolvedShaderUniformValue::Bundle(
                    resolve(ctx.clone(), v.clone()).await?,
                )),
                ShaderUniformValue::Float32Asset(v) => {
                    Ok(ResolvedShaderUniformValue::Float32(*ctx.get(v).await?))
                }
                ShaderUniformValue::Uint32Asset(v) => {
                    Ok(ResolvedShaderUniformValue::Uint32(*ctx.get(v).await?))
                }
                ShaderUniformValue::UsizeAsset(v) => {
                    Ok(ResolvedShaderUniformValue::Usize(*ctx.get(v).await?))
                }
                ShaderUniformValue::TextureAsset(v) => Ok(ResolvedShaderUniformValue::Texture(
                    (*ctx.get(v).await?).clone(),
                )),
                ShaderUniformValue::BufferAsset(v) => Ok(ResolvedShaderUniformValue::Buffer(
                    (*ctx.get(v).await?).clone(),
                )),
                ShaderUniformValue::BundleAsset(v) => Ok(ResolvedShaderUniformValue::Bundle(
                    resolve(ctx.clone(), (*ctx.get(v).await?).clone()).await?,
                )),
            }
        }
            .boxed()
    }
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
    payload: ResolvedShaderUniformPayload,
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
            payload: ResolvedShaderUniformPayload {
                value: self.value.resolve(ctx.clone()).await?,
                warn_if_unreferenced: true,
            },
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
    descriptor_set_layout_info: DescriptorSetLayoutInfo,
    local_size: (u32, u32, u32),
}

unsafe impl Send for ComputeShader {}
unsafe impl Sync for ComputeShader {}

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
    shader_kind: shaderc::ShaderKind,
) -> Result<shaderc::CompilationArtifact> {
    let source = get_shader_text(source);
    shaderc_compile_glsl_str(shader_name, &source, shader_kind)
}

fn shaderc_compile_glsl_str(
    shader_name: &str,
    source: &str,
    shader_kind: shaderc::ShaderKind,
) -> Result<shaderc::CompilationArtifact> {
    let mut compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.add_macro_definition("EP", Some("main"));
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    options.set_generate_debug_info();
    options.set_auto_bind_uniforms(true);
    let binary_result = compiler
        .compile_into_spirv(
            source,
            shader_kind,
            shader_name,
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

#[derive(Default)]
struct DescriptorSetLayoutInfo {
    all_layouts: Vec<vk::DescriptorSetLayout>,
    dynamic_layouts: Vec<vk::DescriptorSetLayout>,
    dynamic_layout_indices: Vec<usize>,
}

impl DescriptorSetLayoutInfo {
    fn append(&mut self, other: &mut Self) {
        let all_layouts_offset = self.all_layouts.len();
        self.all_layouts.append(&mut other.all_layouts);
        self.dynamic_layouts.append(&mut other.dynamic_layouts);
        self.dynamic_layout_indices.extend(
            other
                .dynamic_layout_indices
                .iter()
                .copied()
                .map(|i| i + all_layouts_offset),
        );
    }
}

fn generate_descriptor_set_layouts(
    refl: &spirv_reflect::ShaderModule,
    stage_flags: vk::ShaderStageFlags,
) -> std::result::Result<DescriptorSetLayoutInfo, &'static str> {
    let mut all_layouts = Vec::new();
    let mut is_dynamic = Vec::new();

    let vk = vk();

    let entry = Some("main");
    for descriptor_set in refl.enumerate_descriptor_sets(entry)?.iter() {
        let mut is_set_dynamic = true;

        //println!("{:#?}", descriptor_set);
        let mut binding_flags: Vec<vk::DescriptorBindingFlagsEXT> = Vec::new();
        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = Vec::new();
        let mut immutable_samplers: Vec<vk::Sampler> =
            Vec::with_capacity(descriptor_set.bindings.len());

        let mut create_binding =
            |desc_type,
             binding: &spirv_reflect::types::descriptor::ReflectDescriptorBinding,
             bindings: &mut Vec<vk::DescriptorSetLayoutBinding>,
             binding_flags: &mut Vec<vk::DescriptorBindingFlagsEXT>| {
                let sub_type = *binding.type_description.as_ref().unwrap().op;
                //println!("{:#?}", binding);

                let mut binding_builder = vk::DescriptorSetLayoutBinding::builder()
                    .descriptor_count(binding.count)
                    .descriptor_type(desc_type)
                    .stage_flags(stage_flags)
                    .binding(binding.binding);

                // TODO
                // Bindless resources
                if "all_buffers" == binding.name {
                    is_set_dynamic = false;

                    assert_eq!(desc_type, vk::DescriptorType::UNIFORM_TEXEL_BUFFER);
                    assert_eq!(sub_type, spirv_headers::Op::TypeRuntimeArray);

                    binding_flags.push(
                        vk::DescriptorBindingFlagsEXT::VARIABLE_DESCRIPTOR_COUNT
                            | vk::DescriptorBindingFlagsEXT::PARTIALLY_BOUND
                            | vk::DescriptorBindingFlagsEXT::UPDATE_UNUSED_WHILE_PENDING,
                    );
                    binding_builder = binding_builder.descriptor_count(1 << 18).stage_flags(
                        vk::ShaderStageFlags::COMPUTE
                            | vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT,
                    );
                } else if "all_textures" == binding.name {
                    is_set_dynamic = false;

                    assert_eq!(desc_type, vk::DescriptorType::SAMPLED_IMAGE);
                    assert_eq!(sub_type, spirv_headers::Op::TypeRuntimeArray);

                    binding_flags.push(
                        vk::DescriptorBindingFlagsEXT::VARIABLE_DESCRIPTOR_COUNT
                            | vk::DescriptorBindingFlagsEXT::PARTIALLY_BOUND
                            | vk::DescriptorBindingFlagsEXT::UPDATE_UNUSED_WHILE_PENDING,
                    );
                    binding_builder = binding_builder.descriptor_count(1 << 18).stage_flags(
                        vk::ShaderStageFlags::COMPUTE
                            | vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT,
                    );
                } else {
                    binding_flags.push(vk::DescriptorBindingFlagsEXT::empty());
                }

                bindings.push(binding_builder.build());
            };

        for binding in descriptor_set.bindings.iter() {
            use spirv_reflect::types::descriptor::ReflectDescriptorType;

            match binding.descriptor_type {
                ReflectDescriptorType::UniformBuffer => create_binding(
                    vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    binding,
                    &mut bindings,
                    &mut binding_flags,
                ),
                ReflectDescriptorType::StorageImage => create_binding(
                    vk::DescriptorType::STORAGE_IMAGE,
                    binding,
                    &mut bindings,
                    &mut binding_flags,
                ),
                ReflectDescriptorType::SampledImage => create_binding(
                    vk::DescriptorType::SAMPLED_IMAGE,
                    binding,
                    &mut bindings,
                    &mut binding_flags,
                ),
                ReflectDescriptorType::StorageBuffer => create_binding(
                    vk::DescriptorType::STORAGE_BUFFER,
                    binding,
                    &mut bindings,
                    &mut binding_flags,
                ),
                ReflectDescriptorType::UniformTexelBuffer => create_binding(
                    vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                    binding,
                    &mut bindings,
                    &mut binding_flags,
                ),
                ReflectDescriptorType::Sampler => {
                    immutable_samplers.push(vk.samplers[crate::vulkan::SAMPLER_LINEAR]);
                    binding_flags.push(vk::DescriptorBindingFlagsEXT::empty());
                    bindings.push(
                        vk::DescriptorSetLayoutBinding::builder()
                            .descriptor_count(binding.count)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .stage_flags(stage_flags)
                            .binding(binding.binding)
                            .immutable_samplers(std::slice::from_ref(
                                &immutable_samplers.last().unwrap(),
                            ))
                            .build(),
                    );
                }
                _ => {
                    dbg!(&binding);
                }
            }
        }

        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::builder()
            .binding_flags(&binding_flags)
            .build();

        let descriptor_set_layout = unsafe {
            vk.device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder()
                        .bindings(&bindings)
                        .push_next(&mut binding_flags)
                        .build(),
                    None,
                )
                .unwrap()
        };

        all_layouts.push(descriptor_set_layout);
        is_dynamic.push(is_set_dynamic);
    }

    let mut dynamic_layouts = Vec::new();
    let mut dynamic_layout_indices = Vec::new();

    for (i, l) in all_layouts.iter().copied().enumerate() {
        if is_dynamic[i] {
            dynamic_layouts.push(l);
            dynamic_layout_indices.push(i);
        }
    }

    Ok(DescriptorSetLayoutInfo {
        all_layouts,
        dynamic_layouts,
        dynamic_layout_indices,
    })
}

fn create_compute_pipeline(
    device: &Device,
    descriptor_set_layouts: &[vk::DescriptorSetLayout],
    shader_code: &[u32],
) -> Result<ComputePipeline> {
    use std::ffi::CString;

    let shader_entry_name = CString::new("main").unwrap();

    let layout_create_info =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);

    unsafe {
        let shader_module = device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::builder().code(&shader_code),
                None,
            )
            .unwrap();

        let stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(shader_module)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(&shader_entry_name);

        let pipeline_layout = device
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_create_info.build())
            .layout(pipeline_layout);

        // TODO: pipeline cache
        let pipeline = device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None)
            .expect("pipeline")[0];

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
fn compact_descriptor_sets(refl: &mut spirv_reflect::ShaderModule, set_idx_offset: u32) -> u32 {
    let entry = Some("main");
    let sets = refl
        .enumerate_descriptor_sets(entry)
        .expect("enumerate_descriptor_sets");
    let mut set_order: Vec<_> = sets.iter().enumerate().map(|(i, s)| (i, s.set)).collect();
    set_order.sort_by_key(|(_, set_idx)| *set_idx);
    let set_count = set_order.len();
    for (new_idx, (old_idx, _)) in set_order.into_iter().enumerate() {
        refl.change_descriptor_set_number(&sets[old_idx], new_idx as u32 + set_idx_offset)
            .expect("change_descriptor_set_number");
    }
    set_count as u32
}

fn load_cs_impl(name: String, source: &[shader_prepper::SourceChunk]) -> Result<ComputeShader> {
    let refl = {
        let spirv = shaderc_compile_glsl(&name, source, shaderc::ShaderKind::Compute)?;

        let mut refl = reflect_spirv_shader(spirv.as_binary())?;
        compact_descriptor_sets(&mut refl, 0);
        refl
    };

    let spirv_binary = refl.get_code();

    let local_size = get_cs_local_size_from_spirv(&spirv_binary)?;

    let descriptor_set_layout_info = convert_spirv_reflect_err(generate_descriptor_set_layouts(
        &refl,
        vk::ShaderStageFlags::COMPUTE,
    ))?;

    let vk = vk();
    let pipeline = create_compute_pipeline(
        &vk.device,
        &descriptor_set_layout_info.all_layouts,
        &spirv_binary,
    )?;

    Ok(ComputeShader {
        name,
        pipeline,
        spirv_reflection: refl,
        descriptor_set_layout_info,
        local_size,
    })
}

#[snoozy]
pub async fn load_cs_snoozy(ctx: Context, path: &AssetPath) -> Result<ComputeShader> {
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
pub async fn load_cs_from_string_snoozy(
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
    //module: spirv_reflect::ShaderModule, // Note: spirv_reflect::ShaderModule should not be Clone! It uses a Drop which will corrupt heap if cloned
    spirv: shaderc::CompilationArtifact,
    stage_flags: vk::ShaderStageFlags,
}

unsafe impl Send for RasterSubShader {}
unsafe impl Sync for RasterSubShader {}

impl Drop for RasterSubShader {
    fn drop(&mut self) {
        // TODO: defer
        /*unsafe {
            gl.DeleteShader(self.handle);
        }*/
    }
}

#[snoozy]
pub async fn load_vs_snoozy(ctx: Context, path: &AssetPath) -> Result<RasterSubShader> {
    let source = shader_prepper::process_file(
        &path.asset_name,
        &mut ShaderIncludeProvider { ctx: ctx.clone() },
        AssetPath {
            crate_name: path.crate_name.clone(),
            asset_name: String::new(),
        },
    )?;

    let name = "vs"; // TODO
    let spirv = shaderc_compile_glsl(&name, &source, shaderc::ShaderKind::Vertex)?;

    Ok(RasterSubShader {
        spirv,
        stage_flags: vk::ShaderStageFlags::VERTEX,
    })
}

#[snoozy]
pub async fn load_ps_snoozy(ctx: Context, path: &AssetPath) -> Result<RasterSubShader> {
    let source = shader_prepper::process_file(
        &path.asset_name,
        &mut ShaderIncludeProvider { ctx: ctx.clone() },
        AssetPath {
            crate_name: path.crate_name.clone(),
            asset_name: String::new(),
        },
    )?;

    let name = "ps"; // TODO
    let spirv = shaderc_compile_glsl(&name, &source, shaderc::ShaderKind::Fragment)?;

    Ok(RasterSubShader {
        spirv,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
    })
}

pub struct RasterPipeline {
    pipeline: vk::Pipeline,
    //shaders: Vec<RasterSubShader>,
    shader_refl: Vec<spirv_reflect::ShaderModule>,
    descriptor_set_layout_info: DescriptorSetLayoutInfo,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
}

unsafe impl Send for RasterPipeline {}
unsafe impl Sync for RasterPipeline {}

#[snoozy]
pub async fn make_raster_pipeline_snoozy(
    mut ctx: Context,
    shaders_in: &Vec<SnoozyRef<RasterSubShader>>,
) -> Result<RasterPipeline> {
    use std::ffi::CString;

    let mut shaders = Vec::with_capacity(shaders_in.len());
    for a in shaders_in.iter() {
        shaders.push(ctx.get(&*a).await?);
    }

    let surface_format = vk::Format::R32G32B32A32_SFLOAT;
    //let (width, height) = vk().swapchain_size_pixels();
    let width = 1;
    let height = 1;

    let renderpass_attachments = [
        vk::AttachmentDescription {
            format: surface_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ..Default::default()
        },
        vk::AttachmentDescription {
            format: vk::Format::D24_UNORM_S8_UINT,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            initial_layout: vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
            final_layout: vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
            ..Default::default()
        },
    ];
    let color_attachment_refs = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let depth_attachment_ref = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
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
        .depth_stencil_attachment(&depth_attachment_ref)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .build()];

    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&renderpass_attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    let vk = vk();

    unsafe {
        let render_pass = vk
            .device
            .create_render_pass(&render_pass_create_info, None)
            .unwrap();

        let mut descriptor_set_layout_info = DescriptorSetLayoutInfo::default();
        let mut shader_modules_code = Vec::new();
        let mut shader_refl = Vec::with_capacity(shaders.len());

        // TODO: more efficient concat
        {
            let mut dset_offset = 0u32;
            for s in shaders.iter() {
                let mut refl = reflect_spirv_shader(s.spirv.as_binary())?;
                dset_offset += compact_descriptor_sets(&mut refl, dset_offset);

                let mut shader_descriptor_set_info = convert_spirv_reflect_err(
                    generate_descriptor_set_layouts(&refl, s.stage_flags),
                )?;

                shader_modules_code.push(refl.get_code());
                shader_refl.push(refl);

                descriptor_set_layout_info.append(&mut shader_descriptor_set_info);
            }
        }

        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layout_info.all_layouts)
            .build();
        let pipeline_layout = vk
            .device
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap();

        let shader_entry_name = CString::new("main").unwrap();
        let shader_stage_create_infos: Vec<_> = shaders
            .iter()
            .enumerate()
            .map(|(sub_shader_idx, sub_shader)| {
                let code = &shader_modules_code[sub_shader_idx];
                let shader_info = vk::ShaderModuleCreateInfo::builder().code(code);
                let shader_module = vk
                    .device
                    .create_shader_module(&shader_info, None)
                    .expect("Shader module error");

                vk::PipelineShaderStageCreateInfo {
                    module: shader_module,
                    p_name: shader_entry_name.as_ptr(),
                    stage: sub_shader.stage_flags,
                    ..Default::default()
                }
            })
            .collect();

        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_attribute_description_count: 0,
            p_vertex_attribute_descriptions: std::ptr::null(),
            vertex_binding_description_count: 0,
            p_vertex_binding_descriptions: std::ptr::null(),
            ..Default::default()
        };
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: ash::vk::CullModeFlags::BACK,
            ..Default::default()
        };
        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let noop_stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            ..Default::default()
        };
        let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: 1,
            depth_write_enable: 1,
            depth_compare_op: vk::CompareOp::GREATER_OR_EQUAL,
            front: noop_stencil_state,
            back: noop_stencil_state,
            max_depth_bounds: 1.0,
            ..Default::default()
        };
        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ZERO,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::all(),
        }];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op(vk::LogicOp::CLEAR)
            .attachments(&color_blend_attachment_states);

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_state);

        let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stage_create_infos)
            .vertex_input_state(&vertex_input_state_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_state_info)
            .depth_stencil_state(&depth_state_info)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .render_pass(render_pass);

        let graphics_pipelines = vk
            .device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[graphic_pipeline_info.build()],
                None,
            )
            .expect("Unable to create graphics pipeline");

        let framebuffer = {
            let color_formats = [surface_format];
            let color_attachment = vk::FramebufferAttachmentImageInfoKHR::builder()
                .width(width as _)
                .height(height as _)
                .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
                .layer_count(1)
                .view_formats(&color_formats)
                .usage(
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                )
                .build();
            let depth_attachment = vk::FramebufferAttachmentImageInfoKHR::builder()
                .width(width as _)
                .height(height as _)
                .layer_count(1)
                .view_formats(&[vk::Format::D24_UNORM_S8_UINT])
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .build();
            let attachments = [color_attachment, depth_attachment];
            let mut imageless_desc = vk::FramebufferAttachmentsCreateInfoKHR::builder()
                .attachment_image_infos(&attachments);
            let mut fbo_desc = vk::FramebufferCreateInfo::builder()
                .flags(vk::FramebufferCreateFlags::IMAGELESS_KHR)
                .render_pass(render_pass)
                .width(width as _)
                .height(height as _)
                .layers(1)
                .push_next(&mut imageless_desc);
            fbo_desc.attachment_count = 2;
            vk.device.create_framebuffer(&fbo_desc, None)?
        };

        let graphic_pipeline = graphics_pipelines[0];
        Ok(RasterPipeline {
            pipeline: graphic_pipeline,
            //shaders: shaders,
            shader_refl,
            descriptor_set_layout_info,
            pipeline_layout,
            render_pass,
            framebuffer,
        })
    }
}

pub enum FlattenedUniformEvent {
    SetUniform {
        name: String,
        payload: ResolvedShaderUniformPayload,
    },
    EnterScope,
    LeaveScope,
}

fn flatten_uniforms(
    mut uniforms: Vec<ResolvedShaderUniformHolder>,
    sink: &mut impl FnMut(FlattenedUniformEvent),
) {
    // Do non-bundle values first so that they become visible to bundle handlers
    for uniform in uniforms.iter_mut() {
        let warn_if_unreferenced = uniform.payload.warn_if_unreferenced;

        match uniform.payload.value {
            ResolvedShaderUniformValue::Bundle(_) => {}
            ResolvedShaderUniformValue::Texture(ref value)
            | ResolvedShaderUniformValue::RwTexture(ref value) => {
                let name = std::mem::replace(&mut uniform.name, String::new());

                let tex_size_uniform = (
                    value.key.width as f32,
                    value.key.height as f32,
                    1f32 / value.key.width as f32,
                    1f32 / value.key.height as f32,
                );

                sink(FlattenedUniformEvent::SetUniform {
                    name: name.clone() + "_size",
                    payload: ResolvedShaderUniformPayload {
                        value: ResolvedShaderUniformValue::Vec4(tex_size_uniform),
                        warn_if_unreferenced: false,
                    },
                });

                sink(FlattenedUniformEvent::SetUniform {
                    name,
                    payload: match &uniform.payload.value {
                        ResolvedShaderUniformValue::Texture(ref value) => {
                            ResolvedShaderUniformPayload {
                                value: ResolvedShaderUniformValue::Texture(value.clone()),
                                warn_if_unreferenced,
                            }
                        }
                        ResolvedShaderUniformValue::RwTexture(ref value) => {
                            ResolvedShaderUniformPayload {
                                value: ResolvedShaderUniformValue::RwTexture(value.clone()),
                                warn_if_unreferenced,
                            }
                        }
                        _ => panic!(),
                    },
                });
            }
            _ => {
                let name = std::mem::replace(&mut uniform.name, String::new());
                let value = std::mem::replace(
                    &mut uniform.payload.value,
                    ResolvedShaderUniformValue::Int32(0),
                );

                sink(FlattenedUniformEvent::SetUniform {
                    name,
                    payload: ResolvedShaderUniformPayload {
                        value,
                        warn_if_unreferenced,
                    },
                });
            }
        }
    }

    // Now process bundles
    for uniform in uniforms.into_iter() {
        match uniform.payload.value {
            ResolvedShaderUniformValue::Bundle(bundle) => {
                sink(FlattenedUniformEvent::EnterScope);
                flatten_uniforms(bundle, sink);
                sink(FlattenedUniformEvent::LeaveScope);
            }
            _ => {}
        }
    }
}

struct DescritorSetUpdateResult {
    dynamic_offsets: Vec<u32>,
    all_buffers_descriptor_set_idx: Vec<usize>,
    all_textures_descriptor_set_idx: Vec<usize>,
}

trait DescriptorSetSlice {
    fn get_at_idx(&self, idx: usize) -> std::result::Result<vk::DescriptorSet, &'static str>;
}

impl DescriptorSetSlice for Vec<Option<vk::DescriptorSet>> {
    fn get_at_idx(&self, idx: usize) -> std::result::Result<vk::DescriptorSet, &'static str> {
        match self[idx] {
            Some(d) => Ok(d),
            None => Err("Not a dynamic descriptor set"),
        }
    }
}

trait UniformParamSource {
    fn len(&self) -> usize;
    fn get(&mut self, name: &str) -> Option<&ResolvedShaderUniformValue>;
}

fn update_descriptor_sets<'a>(
    device: &Device,
    refl: impl Iterator<Item = &'a spirv_reflect::ShaderModule>,
    descriptor_sets: &impl DescriptorSetSlice,
    uniforms: &mut impl UniformParamSource,
) -> std::result::Result<DescritorSetUpdateResult, &'static str> {
    use std::cell::RefCell;

    let mut ds_offsets = Vec::new();

    #[derive(Default)]
    pub struct Cache {
        ds_image_info: RefCell<Vec<[vk::DescriptorImageInfo; 1]>>,
        ds_buffer_info: RefCell<Vec<[vk::DescriptorBufferInfo; 1]>>,
        ds_buffer_views: RefCell<Vec<[vk::BufferView; 1]>>,
        ds_writes: RefCell<Vec<vk::WriteDescriptorSet>>,
    }

    // TODO: cache those in thread-locals
    thread_local! {
        pub static CACHE: Cache = Default::default();
    }

    let mut all_buffers_descriptor_set_idx = Vec::new();
    let mut all_textures_descriptor_set_idx = Vec::new();

    CACHE.with(|cache| -> std::result::Result<(), &'static str> {
        let mut ds_image_info = cache.ds_image_info.borrow_mut();
        let mut ds_buffer_info = cache.ds_buffer_info.borrow_mut();
        let mut ds_buffer_views = cache.ds_buffer_views.borrow_mut();
        let mut ds_writes = cache.ds_writes.borrow_mut();

        ds_image_info.clear();
        ds_image_info.reserve(uniforms.len());

        ds_buffer_info.clear();
        ds_buffer_info.reserve(uniforms.len());

        ds_buffer_views.clear();
        ds_buffer_views.reserve(uniforms.len());

        ds_writes.clear();
        ds_writes.reserve(uniforms.len());

        let vk_state = vk_state();
        let vk_frame = vk_state.current_frame();

        for refl in refl {
            let entry = Some("main");
            for descriptor_set in refl.enumerate_descriptor_sets(entry)?.iter() {
                for binding in descriptor_set.bindings.iter() {
                    use spirv_reflect::types::descriptor::ReflectDescriptorType;

                    match binding.descriptor_type {
                        ReflectDescriptorType::UniformBuffer => {
                            let buffer_bytes = binding.block.size as usize;
                            let (buffer_handle, buffer_offset, buffer_contents) = vk_frame
                                .uniforms
                                .allocate(buffer_bytes)
                                .expect("failed to allocate uniform buffer");

                            for member in binding.block.members.iter() {
                                if let Some(value) = uniforms.get(&member.name) {
                                    let dst_mem = &mut buffer_contents[member.absolute_offset
                                        as usize
                                        ..(member.absolute_offset + member.size) as usize];

                                    match value {
                                        ResolvedShaderUniformValue::Float32(value) => {
                                            dst_mem.copy_from_slice(&(*value).to_ne_bytes());
                                        }
                                        ResolvedShaderUniformValue::Uint32(value) => {
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
                                    .dst_set(descriptor_sets.get_at_idx(binding.set as usize)?)
                                    .dst_binding(binding.binding)
                                    .dst_array_element(0)
                                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                                    .buffer_info(buffer_info)
                                    .build(),
                            );
                        }
                        ReflectDescriptorType::SampledImage => {
                            if let Some(ResolvedShaderUniformValue::Texture(value)) =
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
                                        .dst_set(descriptor_sets.get_at_idx(binding.set as usize)?)
                                        .dst_binding(binding.binding)
                                        .dst_array_element(0)
                                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                        .image_info(image_info)
                                        .build(),
                                )
                            } else {
                                if "all_textures" == binding.name {
                                    // Updated elsewhere. Do note the set idx so we can assign the descriptor set afterwards.
                                    all_textures_descriptor_set_idx.push(binding.set as usize);
                                } else {
                                    // TODO
                                    panic!("Could not find resource to bind {}", binding.name);
                                }
                            }
                        }
                        ReflectDescriptorType::StorageImage => {
                            if let Some(ResolvedShaderUniformValue::RwTexture(value)) =
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
                                        .dst_set(descriptor_sets.get_at_idx(binding.set as usize)?)
                                        .dst_binding(binding.binding)
                                        .dst_array_element(0)
                                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                        .image_info(image_info)
                                        .build(),
                                )
                            } else {
                                // TODO
                                panic!("Could not find resource to bind {}", binding.name);
                            }
                        }
                        ReflectDescriptorType::StorageBuffer => {
                            if let Some(ResolvedShaderUniformValue::Buffer(value)) =
                                uniforms.get(&binding.type_description.as_ref().unwrap().type_name)
                            {
                                let buffer_info = [vk::DescriptorBufferInfo::builder()
                                    .buffer(value.buffer)
                                    .range(vk::WHOLE_SIZE)
                                    .build()];
                                ds_buffer_info.push(buffer_info);
                                let buffer_info = ds_buffer_info.last().unwrap();

                                /*println!(
                                    "Updating storage buffer at {}:{}",
                                    binding.set, binding.binding
                                );*/

                                ds_writes.push(
                                    vk::WriteDescriptorSet::builder()
                                        .dst_set(descriptor_sets.get_at_idx(binding.set as usize)?)
                                        .dst_binding(binding.binding)
                                        .dst_array_element(0)
                                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                        .buffer_info(buffer_info)
                                        .build(),
                                );
                            } else {
                                panic!(binding
                                    .type_description
                                    .as_ref()
                                    .unwrap()
                                    .type_name
                                    .clone());
                            }
                        }
                        ReflectDescriptorType::Sampler => {
                            // Immutable. Nothing to do.
                            /*println!(
                                "Skipping sampler update at {}:{}",
                                binding.set, binding.binding
                            );*/
                        }
                        ReflectDescriptorType::UniformTexelBuffer => {
                            if let Some(ResolvedShaderUniformValue::Buffer(value)) =
                                uniforms.get(&binding.name)
                            {
                                ds_buffer_views.push([value.view]);
                                let buffer_view = ds_buffer_views.last().unwrap();

                                ds_writes.push(
                                    vk::WriteDescriptorSet::builder()
                                        .dst_set(descriptor_sets.get_at_idx(binding.set as usize)?)
                                        .dst_binding(binding.binding)
                                        .dst_array_element(0)
                                        .descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                                        .texel_buffer_view(buffer_view)
                                        .build(),
                                );
                            } else {
                                if "all_buffers" == binding.name {
                                    // Updated elsewhere. Do note the set idx so we can assign the descriptor set afterwards.
                                    all_buffers_descriptor_set_idx.push(binding.set as usize);
                                } else {
                                    // TODO
                                    panic!("Could not find resource to bind {}", binding.name);
                                }
                            }
                        }
                        _ => {
                            dbg!(&binding);
                        }
                    }
                }
            }
        }

        //dbg!(&ds_writes);
        //dbg!(&descriptor_sets);
        if !ds_writes.is_empty() {
            unsafe { device.update_descriptor_sets(&ds_writes, &[]) };
        }

        Ok(())
    })?;

    Ok(DescritorSetUpdateResult {
        dynamic_offsets: ds_offsets,
        all_buffers_descriptor_set_idx,
        all_textures_descriptor_set_idx,
    })
}

pub struct ResolvedShaderUniformPayload {
    value: ResolvedShaderUniformValue,
    warn_if_unreferenced: bool,
}

// Provide access to uniforms, but also record which ones are being requested,
// so that we can issue warnings about any unreferenced ones.
struct TrackedUniformParamSource {
    uniforms: HashMap<String, ResolvedShaderUniformPayload>,
    requested: HashSet<String>,
}

impl UniformParamSource for TrackedUniformParamSource {
    fn len(&self) -> usize {
        self.uniforms.len()
    }

    fn get(&mut self, name: &str) -> Option<&ResolvedShaderUniformValue> {
        self.requested.insert(name.to_owned());
        self.uniforms.get(name).map(|v| &v.value)
    }
}

impl TrackedUniformParamSource {
    fn report_unreferenced_uniform_warnings(self, shader_name: &str) {
        let requested = self.requested;
        let unreferenced = self
            .uniforms
            .into_iter()
            .filter_map(move |(name, payload)| {
                if !payload.warn_if_unreferenced || requested.contains(&name) {
                    None
                } else {
                    Some(name)
                }
            });

        for name in unreferenced {
            crate::rtoy_show_warning(format!("Unreferenced uniform {}::{}", shader_name, name));
        }
    }
}

#[snoozy]
pub async fn compute_tex_snoozy(
    mut ctx: Context,
    key: &TextureKey,
    cs: &SnoozyRef<ComputeShader>,
    uniforms: &Vec<ShaderUniformHolder>,
) -> Result<Texture> {
    let output_tex = crate::backend::texture::create_texture(*key);
    let cs = ctx.get(cs).await?;
    ctx.set_debug_name(&cs.name);

    let mut uniforms = resolve(ctx, uniforms.clone()).await?;
    uniforms.push(ResolvedShaderUniformHolder {
        name: "outputTex".to_owned(),
        payload: ResolvedShaderUniformPayload {
            value: ResolvedShaderUniformValue::RwTexture(output_tex.clone()),
            warn_if_unreferenced: true,
        },
    });

    let (vk, vk_state) = vk_all();
    let vk_frame = vk_state.current_frame();

    let mut flattened_uniforms: HashMap<String, ResolvedShaderUniformPayload> = HashMap::new();
    flatten_uniforms(uniforms, &mut |e| {
        if let FlattenedUniformEvent::SetUniform { name, payload } = e {
            flattened_uniforms.insert(name, payload);
        }
    });

    let mut uniform_source = TrackedUniformParamSource {
        uniforms: flattened_uniforms,
        requested: HashSet::new(),
    };

    let (descriptor_sets, ds_update_result) = unsafe {
        let descriptor_sets = {
            let layout_info = &cs.descriptor_set_layout_info;
            let descriptor_pool = vk_frame.descriptor_pool.lock().unwrap();
            let dynamic_sets = vk.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&layout_info.dynamic_layouts)
                    .build(),
            )?;
            drop(descriptor_pool);

            let mut sets = vec![None; layout_info.all_layouts.len()];
            for (src, dst) in layout_info.dynamic_layout_indices.iter().enumerate() {
                sets[*dst] = Some(dynamic_sets[src]);
            }

            sets
        };

        let ds_update_result = update_descriptor_sets(
            &vk.device,
            std::iter::once(&cs.spirv_reflection),
            &descriptor_sets,
            &mut uniform_source,
        )
        .unwrap();

        (descriptor_sets, ds_update_result)
    };

    let cb = vk_frame.command_buffer.lock().unwrap();
    let cb: vk::CommandBuffer = cb.cb;

    unsafe {
        record_image_barrier(
            &vk.device,
            cb,
            ImageBarrier::new(
                output_tex.image,
                vk_sync::AccessType::Nothing,
                vk_sync::AccessType::ComputeShaderWrite,
            )
            .with_discard(true),
        );

        let mut descriptor_sets = descriptor_sets;

        for idx in ds_update_result.all_buffers_descriptor_set_idx.iter() {
            descriptor_sets[*idx] = Some(vk_state.bindless_buffers_descriptor_set);
        }

        for idx in ds_update_result.all_textures_descriptor_set_idx.iter() {
            descriptor_sets[*idx] = Some(vk_state.bindless_images_descriptor_set);
        }

        let descriptor_sets: Vec<_> = descriptor_sets.into_iter().map(Option::unwrap).collect();

        vk.device
            .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, cs.pipeline.pipeline);
        vk.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            cs.pipeline.pipeline_layout,
            0,
            &descriptor_sets,
            &ds_update_result.dynamic_offsets,
        );

        let query_id = crate::gpu_profiler::create_gpu_query(&cs.name);
        let vk_query_idx = vk_frame.profiler_data.get_query_id(query_id);

        vk.device.cmd_write_timestamp(
            cb,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk_frame.profiler_data.query_pool,
            vk_query_idx * 2 + 0,
        );

        vk.device.cmd_dispatch(
            cb,
            (key.width + cs.local_size.0 - 1) / cs.local_size.0,
            (key.height + cs.local_size.1 - 1) / cs.local_size.1,
            1,
        );

        vk.device.cmd_write_timestamp(
            cb,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk_frame.profiler_data.query_pool,
            vk_query_idx * 2 + 1,
        );

        record_image_barrier(
            &vk.device,
            cb,
            ImageBarrier::new(
                output_tex.image,
                vk_sync::AccessType::ComputeShaderWrite,
                vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
            ),
        );
    }

    uniform_source.report_unreferenced_uniform_warnings(&cs.name);
    gpu_debugger::report_texture(&cs.name, output_tex.view);

    Ok(output_tex)
}

#[snoozy]
pub async fn raster_tex_snoozy(
    mut ctx: Context,
    key: &TextureKey,
    raster_pipe: &SnoozyRef<RasterPipeline>,
    uniforms: &Vec<ShaderUniformHolder>,
) -> Result<Texture> {
    let output_tex = crate::backend::texture::create_texture(*key);
    let raster_pipe = ctx.get(raster_pipe).await?;

    let mut uniforms = resolve(ctx.clone(), uniforms.clone()).await?;
    uniforms.push(ResolvedShaderUniformHolder {
        name: "outputTex".to_owned(),
        payload: ResolvedShaderUniformPayload {
            value: ResolvedShaderUniformValue::RwTexture(output_tex.clone()),
            warn_if_unreferenced: false,
        },
    });

    //println!("---- raster_tex: ----");

    let (vk, vk_state) = vk_all();
    let vk_frame = vk_state.current_frame();

    let cb = vk_frame.command_buffer.lock().unwrap();
    let cb: vk::CommandBuffer = cb.cb;

    unsafe {
        record_image_barrier(
            &vk.device,
            cb,
            ImageBarrier::new(
                output_tex.image,
                vk_sync::AccessType::Nothing,
                vk_sync::AccessType::ColorAttachmentWrite,
            )
            .with_discard(true),
        );

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            },
        ];

        let texture_attachments = [output_tex.rt_view, vk_state.depth_image_view];
        let mut pass_attachment_desc =
            vk::RenderPassAttachmentBeginInfoKHR::builder().attachments(&texture_attachments);

        const USE_IMAGELESS: bool = false;

        let framebuffer = if USE_IMAGELESS {
            raster_pipe.framebuffer
        } else {
            // HACK; must not do this, but validation layers are broken with IMAGELESS_KHR
            let fbo_desc = vk::FramebufferCreateInfo::builder()
                .render_pass(raster_pipe.render_pass)
                .width(key.width as _)
                .height(key.height as _)
                .layers(1)
                .attachments(&texture_attachments);
            let fbo = vk.device.create_framebuffer(&fbo_desc, None)?;

            vk_frame
                .frame_cleanup
                .lock()
                .unwrap()
                .push(Box::new(move |vk| {
                    vk.device.destroy_framebuffer(fbo, None);
                }));

            fbo
        };

        let mut pass_begin_desc = vk::RenderPassBeginInfo::builder()
            .render_pass(raster_pipe.render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: key.width as _,
                    height: key.height as _,
                },
            })
            .clear_values(&clear_values);

        if USE_IMAGELESS {
            pass_begin_desc = pass_begin_desc.push_next(&mut pass_attachment_desc)
        }

        vk.device
            .cmd_begin_render_pass(cb, &pass_begin_desc, vk::SubpassContents::INLINE);
    }

    let flush_draw = |uniform_source: &mut TrackedUniformParamSource| -> Result<()> {
        unsafe {
            let (descriptor_sets, ds_update_result) = {
                let descriptor_sets = {
                    let layout_info = &raster_pipe.descriptor_set_layout_info;
                    let descriptor_pool = vk_frame.descriptor_pool.lock().unwrap();
                    let dynamic_sets = vk.device.allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::builder()
                            .descriptor_pool(*descriptor_pool)
                            .set_layouts(&layout_info.dynamic_layouts)
                            .build(),
                    )?;
                    drop(descriptor_pool);

                    let mut sets = vec![None; layout_info.all_layouts.len()];
                    for (src, dst) in layout_info.dynamic_layout_indices.iter().enumerate() {
                        sets[*dst] = Some(dynamic_sets[src]);
                    }

                    sets
                };

                let ds_update_result = update_descriptor_sets(
                    &vk.device,
                    raster_pipe.shader_refl.iter(),
                    &descriptor_sets,
                    uniform_source,
                )
                .unwrap();

                (descriptor_sets, ds_update_result)
            };

            let mut descriptor_sets = descriptor_sets;

            for idx in ds_update_result.all_buffers_descriptor_set_idx.iter() {
                descriptor_sets[*idx] = Some(vk_state.bindless_buffers_descriptor_set);
            }

            for idx in ds_update_result.all_textures_descriptor_set_idx.iter() {
                descriptor_sets[*idx] = Some(vk_state.bindless_images_descriptor_set);
            }

            let descriptor_sets: Vec<_> = descriptor_sets.into_iter().map(Option::unwrap).collect();

            vk.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, raster_pipe.pipeline);

            vk.device.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: (key.height as f32),
                    width: key.width as _,
                    height: -(key.height as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            vk.device.cmd_set_scissor(
                cb,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: key.width as _,
                        height: key.height as _,
                    },
                }],
            );

            vk.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                raster_pipe.pipeline_layout,
                0,
                &descriptor_sets,
                &ds_update_result.dynamic_offsets,
            );

            Ok(())
        }
    };

    #[derive(Default)]
    struct MeshDrawData {
        index_buffer: Option<vk::Buffer>,
        index_count: Option<u32>,
    }

    let mut mesh_stack = vec![MeshDrawData::default()];

    let flattened_uniforms: HashMap<String, ResolvedShaderUniformPayload> = HashMap::new();
    let mut uniform_source = TrackedUniformParamSource {
        uniforms: flattened_uniforms,
        requested: HashSet::new(),
    };

    flatten_uniforms(uniforms, &mut |e| match e {
        FlattenedUniformEvent::SetUniform { name, mut payload } => {
            match payload.value {
                ResolvedShaderUniformValue::Buffer(ref buf) if name == "mesh_index_buf" => {
                    mesh_stack.last_mut().unwrap().index_buffer = Some(buf.buffer);
                    payload.warn_if_unreferenced = false;
                }
                ResolvedShaderUniformValue::Uint32(value) if name == "mesh_index_count" => {
                    mesh_stack.last_mut().unwrap().index_count = Some(value);
                    payload.warn_if_unreferenced = false;
                }
                _ => {}
            }

            uniform_source.uniforms.insert(name, payload);
        }
        FlattenedUniformEvent::EnterScope => {
            mesh_stack.push(Default::default());
        }
        FlattenedUniformEvent::LeaveScope => {
            let mesh = mesh_stack.pop().unwrap();
            if let Some(index_count) = mesh.index_count {
                if let Some(index_buffer) = mesh.index_buffer {
                    unsafe {
                        flush_draw(&mut uniform_source).expect("flush_draw");
                        vk.device
                            .cmd_bind_index_buffer(cb, index_buffer, 0, vk::IndexType::UINT32);
                        vk.device.cmd_draw_indexed(cb, index_count as _, 1, 0, 0, 0);
                        //println!("-------");
                    }
                }
            }
        }
    });

    unsafe {
        vk.device.cmd_end_render_pass(cb);

        record_image_barrier(
            &vk.device,
            cb,
            ImageBarrier::new(
                output_tex.image,
                vk_sync::AccessType::ColorAttachmentWrite,
                vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
            ),
        );
    };

    uniform_source.report_unreferenced_uniform_warnings("mesh_raster");
    gpu_debugger::report_texture("mesh_raster", output_tex.view);

    Ok(output_tex)
}
