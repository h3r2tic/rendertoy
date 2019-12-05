use super::*;
use futures::future::try_join_all;

#[derive(Clone, Abomonation)]
pub enum MeshMaterialMap {
    Asset { path: AssetPath, params: TexParams },
    Placeholder([u8; 4]),
}

#[derive(Clone, Abomonation)]
pub struct MeshMaterial {
    pub base_color_mult: [f32; 4],
    pub maps: [u32; 3],
    pub emissive: [f32; 3],
}

#[derive(Abomonation, Default)]
pub struct TriangleMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
    pub material_ids: Vec<u32>, // per index, but can be flat shaded
    pub indices: Vec<u32>,
    pub materials: Vec<MeshMaterial>, // global
    pub maps: Vec<MeshMaterialMap>,   // global
}

fn iter_gltf_node_tree<F: FnMut(&gltf::scene::Node, Matrix4)>(
    node: &gltf::scene::Node,
    xform: Matrix4,
    f: &mut F,
) {
    let node_xform: Matrix4 = node.transform().matrix().into();
    let xform = xform * node_xform;

    f(&node, xform);
    for child in node.children() {
        iter_gltf_node_tree(&child, xform, f);
    }
}

fn get_gltf_texture_source(tex: gltf::texture::Texture) -> Option<String> {
    match tex.source().source() {
        gltf::image::Source::Uri { uri, .. } => Some(uri.to_string()),
        _ => None,
    }
}

fn load_gltf_material(
    mat: &gltf::material::Material,
    parent_path: &AssetPath,
) -> (Vec<MeshMaterialMap>, MeshMaterial) {
    let make_asset_path = |path: String| -> AssetPath {
        let mut asset_name: std::path::PathBuf = parent_path.asset_name.clone().into();
        asset_name.pop();
        asset_name.push(&path);
        AssetPath {
            crate_name: parent_path.crate_name.clone(),
            asset_name: asset_name.to_string_lossy().to_string(),
        }
    };

    let make_material_map = |path: String| -> MeshMaterialMap {
        MeshMaterialMap::Asset {
            path: make_asset_path(path),
            params: TexParams {
                gamma: TexGamma::Linear,
            },
        }
    };

    let albedo_map = mat
        .pbr_metallic_roughness()
        .base_color_texture()
        .and_then(|tex| {
            get_gltf_texture_source(tex.texture()).map(|path: String| -> MeshMaterialMap {
                MeshMaterialMap::Asset {
                    path: make_asset_path(path),
                    params: TexParams {
                        gamma: TexGamma::Srgb,
                    },
                }
            })
        })
        .unwrap_or(MeshMaterialMap::Placeholder([127, 127, 127, 255]));

    let normal_map = mat
        .normal_texture()
        .and_then(|tex| get_gltf_texture_source(tex.texture()).map(make_material_map))
        .unwrap_or(MeshMaterialMap::Placeholder([127, 127, 255, 255]));

    let spec_map = mat
        .pbr_metallic_roughness()
        .metallic_roughness_texture()
        .and_then(|tex| get_gltf_texture_source(tex.texture()).map(make_material_map))
        .unwrap_or(MeshMaterialMap::Placeholder([127, 127, 0, 255]));

    let emissive = if mat.emissive_texture().is_some() {
        [0.0, 0.0, 0.0]
    } else {
        mat.emissive_factor()
    };

    let base_color_mult = mat.pbr_metallic_roughness().base_color_factor();

    (
        vec![normal_map, spec_map, albedo_map],
        MeshMaterial {
            base_color_mult,
            maps: [0, 1, 2],
            emissive,
        },
    )
}

#[snoozy]
pub async fn load_gltf_scene(ctx: Context, path: &AssetPath, scale: &f32) -> Result<TriangleMesh> {
    let (gltf, buffers, _imgs) = gltf::import(path.to_path_lossy(ctx).await?)?;

    if let Some(scene) = gltf.default_scene() {
        let mut res: TriangleMesh = TriangleMesh::default();

        let mut process_node = |node: &gltf::scene::Node, xform: Matrix4| {
            if let Some(mesh) = node.mesh() {
                for prim in mesh.primitives() {
                    let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

                    let res_material_index = res.materials.len() as u32;

                    {
                        let (mut maps, mut material) = load_gltf_material(&prim.material(), path);

                        let map_base = res.maps.len() as u32;
                        for id in material.maps.iter_mut() {
                            *id += map_base;
                        }

                        res.materials.push(material);
                        res.maps.append(&mut maps);
                    }

                    // Collect positions (required)
                    let positions = if let Some(iter) = reader.read_positions() {
                        iter.collect::<Vec<_>>()
                    } else {
                        return;
                    };

                    // Collect normals (required)
                    let normals = if let Some(iter) = reader.read_normals() {
                        iter.collect::<Vec<_>>()
                    } else {
                        return;
                    };

                    // Collect tangents (optional)
                    let mut tangents = if let Some(iter) = reader.read_tangents() {
                        iter.collect::<Vec<_>>()
                    } else {
                        vec![[1.0, 0.0, 0.0, 0.0]; positions.len()]
                    };

                    // Collect uvs (optional)
                    let mut uvs = if let Some(iter) = reader.read_tex_coords(0) {
                        iter.into_f32().collect::<Vec<_>>()
                    } else {
                        vec![[0.0, 0.0]; positions.len()]
                    };

                    // Collect colors (optional)
                    let colors = if let Some(iter) = reader.read_colors(0) {
                        iter.into_rgba_f32().collect::<Vec<_>>()
                    } else {
                        vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]
                    };

                    // Collect material ids
                    let mut material_ids = vec![res_material_index; positions.len()];

                    // --------------------------------------------------------
                    // Write it all to the output

                    {
                        let mut indices: Vec<u32>;
                        let base_index = res.positions.len() as u32;

                        if let Some(indices_reader) = reader.read_indices() {
                            indices = indices_reader.into_u32().map(|i| i + base_index).collect();
                        } else {
                            indices = (base_index..(base_index + positions.len() as u32)).collect();
                        }

                        res.indices.append(&mut indices);
                        res.tangents.append(&mut tangents);
                        res.material_ids.append(&mut material_ids);
                    }

                    for p in positions {
                        let pos =
                            Point3::from_homogeneous(xform * Point3::from(p).to_homogeneous())
                                .unwrap();
                        res.positions.push([pos.x, pos.y, pos.z]);
                    }

                    for n in normals {
                        let norm =
                            Vector3::from_homogeneous(xform * Vector3::from(n).to_homogeneous())
                                .unwrap()
                                .normalize();
                        res.normals.push([norm.x, norm.y, norm.z]);
                    }

                    for c in colors {
                        res.colors.push(c);
                    }

                    res.uvs.append(&mut uvs);
                }
            }
        };

        let xform = Matrix4::new_scaling(*scale);
        for node in scene.nodes() {
            iter_gltf_node_tree(&node, xform, &mut process_node);
        }

        Ok(res)
    } else {
        Err(format_err!("No default scene found in gltf"))
    }
}

#[derive(Clone, Copy, Abomonation)]
#[repr(C)]
pub struct RasterGpuVertex {
    pos: [f32; 3],
    normal: u32,
}

fn pack_unit_direction_11_10_11(x: f32, y: f32, z: f32) -> u32 {
    let x = ((x.max(-1.0).min(1.0) * 0.5 + 0.5) * ((1u32 << 11u32) - 1u32) as f32) as u32;
    let y = ((y.max(-1.0).min(1.0) * 0.5 + 0.5) * ((1u32 << 10u32) - 1u32) as f32) as u32;
    let z = ((z.max(-1.0).min(1.0) * 0.5 + 0.5) * ((1u32 << 11u32) - 1u32) as f32) as u32;

    (z << 21) | (y << 11) | x
}

#[derive(Clone, Abomonation)]
pub struct RasterGpuMesh {
    verts: Vec<RasterGpuVertex>,
    uvs: Vec<[f32; 2]>,
    tangents: Vec<[f32; 4]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
    material_ids: Vec<u32>,
    materials: Vec<MeshMaterial>,
    maps: Vec<MeshMaterialMap>,
}

#[snoozy]
pub async fn make_raster_mesh(
    ctx: Context,
    mesh: &SnoozyRef<TriangleMesh>,
) -> Result<RasterGpuMesh> {
    let mesh = ctx.get(mesh).await?;

    let mut verts: Vec<RasterGpuVertex> = Vec::with_capacity(mesh.positions.len());

    for (i, pos) in mesh.positions.iter().enumerate() {
        let n = mesh.normals[i];

        verts.push(RasterGpuVertex {
            pos: *pos,
            normal: pack_unit_direction_11_10_11(n[0], n[1], n[2]),
        });
    }

    Ok(RasterGpuMesh {
        verts,
        uvs: mesh.uvs.clone(),
        tangents: mesh.tangents.clone(),
        colors: mesh.colors.clone(),
        indices: mesh.indices.clone(),
        material_ids: mesh.material_ids.clone(),
        materials: mesh.materials.clone(),
        maps: mesh.maps.clone(),
    })
}

#[derive(Copy, Clone, Abomonation, Serialize)]
#[repr(C)]
struct GpuMaterial {
    base_color_mult: [f32; 4],
    maps: [u32; 4],
}

impl Default for GpuMaterial {
    fn default() -> Self {
        Self {
            base_color_mult: [0.0f32; 4],
            maps: [0; 4],
        }
    }
}

async fn upload_material_map(ctx: Context, map: MeshMaterialMap) -> u32 {
    let tex = match map {
        MeshMaterialMap::Asset {
            ref path,
            ref params,
        } => load_tex_with_params(path.clone(), params.clone()),
        MeshMaterialMap::Placeholder(ref texel_value) => make_placeholder_rgba8_tex(*texel_value),
    };

    if let Ok(tex) = ctx.get(tex).await {
        tex.bindless_index
    } else {
        0u32
    }
}

#[snoozy]
pub async fn upload_raster_mesh(
    ctx: Context,
    mesh: &SnoozyRef<RasterGpuMesh>,
) -> Result<ShaderUniformBundle> {
    let mesh = ctx.get(mesh).await?;

    let verts = ArcView::new(&mesh, |m| &m.verts);
    let uvs = ArcView::new(&mesh, |m| &m.uvs);
    let colors = ArcView::new(&mesh, |m| &m.colors);
    let tangents = ArcView::new(&mesh, |m| &m.tangents);
    let indices = ArcView::new(&mesh, |m| &m.indices);
    let material_ids = ArcView::new(&mesh, |m| &m.material_ids);

    // TODO: don't clone all the things
    let (mesh_materials, mesh_maps) = {
        let mesh_clone = (*mesh).clone();
        (mesh_clone.materials, mesh_clone.maps)
    };

    let materials: Vec<GpuMaterial> = try_join_all(mesh_materials.into_iter().map(|m| {
        let ctx = ctx.clone();

        // TODO: don't clone all the things
        let mesh_maps = mesh_maps.clone();
        tokio::task::spawn(async move {
            let mut res = GpuMaterial::default();
            res.base_color_mult = m.base_color_mult;

            let maps: Vec<_> = try_join_all(m.maps.iter().map(|map_id| {
                tokio::task::spawn(
                    // TODO: don't clone all the things
                    upload_material_map(ctx.clone(), mesh_maps[*map_id as usize].clone()),
                )
            }))
            .await
            .expect("tokio join error");

            for (i, m) in maps.into_iter().enumerate() {
                res.maps[i] = m;
            }

            res
        })
    }))
    .await
    .expect("tokio join error");

    Ok(shader_uniforms!(
        mesh_vertex_buf: upload_array_buffer(verts),
        mesh_uv_buf: upload_array_buffer(uvs),
        mesh_color_buf: upload_array_buffer(colors),
        mesh_tangent_buf: upload_array_buffer(tangents),
        mesh_index_count: indices.len() as u32,
        mesh_index_buf: upload_array_buffer(indices),
        mesh_material_id_buf: upload_array_buffer(material_ids),
        mesh_materials_buf: upload_array_buffer(Box::new(materials)),
    ))
}

pub fn raster_mesh_transform(offset: Vector3, rotation: UnitQuaternion) -> SnoozyRef<Buffer> {
    let model_to_world: Matrix4 = {
        let translation = Matrix4::new_translation(&Vector3::new(offset.x, offset.y, offset.z));
        translation * rotation.to_homogeneous()
    };

    upload_buffer(model_to_world)
}

pub fn upload_raster_scene(
    scene: &[(SnoozyRef<TriangleMesh>, Vector3, UnitQuaternion)],
) -> ShaderUniformBundle {
    scene
        .iter()
        .map(|(mesh, position, rotation)| {
            shader_uniform_bundle!(
                instance_transform: raster_mesh_transform(*position, *rotation),
                :upload_raster_mesh(make_raster_mesh(mesh.clone()))
            )
        })
        .collect()
}
