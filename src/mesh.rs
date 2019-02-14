use super::*;

use obj::raw::object::Polygon;
use obj::*;

/*
use std::io;

pub struct Mesh {
    #[allow(dead_code)]
    models: Vec<tobj::Model>,
}

snoozy! {
    fn load_mesh(ctx: &mut Context, blob: &SnoozyRef<Blob>) -> Result<Mesh> {
        let m = tobj::load_obj_buf(&mut io::Cursor::new(&ctx.get(blob)?.contents), |_| {
            Ok((Default::default(), Default::default()))
        })?;

        let (models, _materials) = m;
        Ok(Mesh { models: models })
    }
}
*/

#[derive(Serialize)]
pub struct Triangle {
    pub a: Point3,
    pub b: Point3,
    pub c: Point3,
}

impl Triangle {
    pub fn new(a: Point3, b: Point3, c: Point3) -> Triangle {
        Triangle { a: a, b: b, c: c }
    }
}

impl FromRawVertex for Triangle {
    fn process(
        vertices: Vec<(f32, f32, f32, f32)>,
        _: Vec<(f32, f32, f32)>,
        polygons: Vec<Polygon>,
    ) -> ObjResult<(Vec<Self>, Vec<u16>)> {
        // Convert the vertices to `Point3`s.
        let points = vertices
            .into_iter()
            .map(|v| Point3::new(v.0, v.1, v.2))
            .collect::<Vec<_>>();

        // Estimate for the number of triangles, assuming that each polygon is a triangle.
        let mut triangles = Vec::with_capacity(polygons.len());
        {
            let mut push_triangle = |indices: &Vec<usize>| {
                let mut indices_iter = indices.iter();
                let anchor = points[*indices_iter.next().unwrap()];
                let mut second = points[*indices_iter.next().unwrap()];
                for third_index in indices_iter {
                    let third = points[*third_index];
                    triangles.push(Triangle::new(anchor, second, third));
                    second = third;
                }
            };

            // Iterate over the polygons and populate the `Triangle`s vector.
            for polygon in polygons.into_iter() {
                match polygon {
                    Polygon::P(ref vec) => push_triangle(vec),
                    Polygon::PT(ref vec) | Polygon::PN(ref vec) => {
                        push_triangle(&vec.iter().map(|vertex| vertex.0).collect())
                    }
                    Polygon::PTN(ref vec) => {
                        push_triangle(&vec.iter().map(|vertex| vertex.0).collect())
                    }
                }
            }
        }
        Ok((triangles, Vec::new()))
    }
}

snoozy! {
    fn load_obj_scene(_ctx: &mut Context, path: &String) -> Result<Vec<Triangle>> {
        use libflate::gzip::Decoder;
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        use std::path::Path;

        let f = BufReader::new(File::open(path)?);

        let f: Box<dyn BufRead> = if Path::new(path).extension().unwrap() == "gz" {
            let f = Decoder::new(f).unwrap();
            Box::new(std::io::BufReader::new(f))
        } else {
            Box::new(f)
        };

        let obj: Obj<Triangle> = load_obj(f)?;

        Ok(obj.vertices)
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct RasterGpuVertex {
    pos: [f32; 3],
    normal: u32,
}

fn pack_unit_direction_11_10_11(x: f32, y: f32, z: f32) -> u32 {
    let x = ((x.max(-1.0).min(1.0) * 0.5 + 0.5) * ((1u32 << 11u32) - 1u32) as f32) as u32;
    let y = ((y.max(-1.0).min(1.0) * 0.5 + 0.5) * ((1u32 << 10u32) - 1u32) as f32) as u32;
    let z = ((z.max(-1.0).min(1.0) * 0.5 + 0.5) * ((1u32 << 11u32) - 1u32) as f32) as u32;

    (z << 21) | (y << 11) | x
}

snoozy! {
    fn make_raster_mesh(
        ctx: &mut Context,
        mesh: &SnoozyRef<Vec<Triangle>>
    ) -> Result<ShaderUniformBundle> {
        let mesh = ctx.get(mesh)?;
        let mut verts: Vec<RasterGpuVertex> = Vec::with_capacity(mesh.len() * 3);

        for tri in mesh.iter() {
            let e0 = tri.b - tri.a;
            let e1 = tri.c - tri.a;
            let n = e0.cross(&e1).normalize();

            for v in &[&tri.a, &tri.b, &tri.c] {
                verts.push(RasterGpuVertex {
                    pos: [v.x, v.y, v.z],
                    normal: pack_unit_direction_11_10_11(n.x, n.y, n.z),
                });
            }
        }

        Ok(shader_uniforms!(
            "mesh_vertex_buf": upload_buffer(to_byte_vec(verts)),
            "mesh_index_count": (mesh.len() * 3) as u32
        ))
    }
}
