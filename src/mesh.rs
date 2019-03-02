use super::*;

#[derive(Abomonation, Default)]
pub struct TriangleMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
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

#[snoozy]
pub fn load_gltf_scene(_ctx: &mut Context, path: &String, scale: &f32) -> Result<TriangleMesh> {
    let (gltf, buffers, _images) = gltf::import(path)?;

    if let Some(scene) = gltf.default_scene() {
        let mut res: TriangleMesh = TriangleMesh::default();

        let mut process_node = |node: &gltf::scene::Node, xform: Matrix4| {
            if let Some(mesh) = node.mesh() {
                for prim in mesh.primitives() {
                    let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

                    let positions = if let Some(iter) = reader.read_positions() {
                        iter.collect::<Vec<_>>()
                    } else {
                        return;
                    };

                    let normals = if let Some(iter) = reader.read_normals() {
                        iter.collect::<Vec<_>>()
                    } else {
                        return;
                    };

                    {
                        let mut indices: Vec<u32>;
                        let base_index = res.positions.len() as u32;

                        if let Some(indices_reader) = reader.read_indices() {
                            indices = indices_reader.into_u32().map(|i| i + base_index).collect();
                        } else {
                            indices = (base_index..(base_index + positions.len() as u32)).collect();
                        }

                        res.indices.append(&mut indices);
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
    indices: Vec<u32>,
}

#[snoozy]
pub fn make_raster_mesh(
    ctx: &mut Context,
    mesh: &SnoozyRef<TriangleMesh>,
) -> Result<RasterGpuMesh> {
    let mesh = ctx.get(mesh)?;

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
        indices: mesh.indices.clone(),
    })
}

#[snoozy]
pub fn upload_raster_mesh(
    ctx: &mut Context,
    mesh: &SnoozyRef<RasterGpuMesh>,
) -> Result<ShaderUniformBundle> {
    let mesh = ctx.get(mesh)?;

    let verts = ArcView::new(&mesh, |m| &m.verts);
    let indices = ArcView::new(&mesh, |m| &m.indices);

    Ok(shader_uniforms!(
        "mesh_vertex_buf": upload_array_buffer(verts),
        "mesh_index_count": indices.len() as u32,
        "mesh_index_buf": upload_array_buffer(indices),
    ))
}
