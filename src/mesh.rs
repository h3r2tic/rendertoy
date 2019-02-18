use super::*;

use obj::raw::object::Polygon;
use obj::*;

use kdtree::distance::squared_euclidean;
use kdtree::KdTree;

#[derive(Serialize, Deserialize)]
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

#[snoozy]
pub fn load_obj_scene(_ctx: &mut Context, path: &String) -> Result<Vec<Triangle>> {
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
    index_count: u32,
}

#[derive(Clone, Copy)]
pub struct TriangleNormals {
    pub a: Vector3,
    pub b: Vector3,
    pub c: Vector3,
}

fn calculate_vertex_normals(mesh: &Vec<Triangle>) -> Vec<TriangleNormals> {
    let mut kdtree = KdTree::new(3);

    let eps = 1e-3_f32;
    let eps2 = eps * eps;

    let get_equivalent = |kdtree: &KdTree<_, _, _>, pt: [f32; 3]| {
        for (nearest_dist2, nearest_i) in kdtree.iter_nearest(&pt, &squared_euclidean).unwrap() {
            // TODO: normal cutoff
            if nearest_dist2 < eps2 {
                return Some(*nearest_i);
            } else {
                break;
            }
        }
        None
    };

    let mut vert_normals = vec![Vector3::zeros(); mesh.len() * 3];

    for (tri_i, tri) in mesh.iter().enumerate() {
        let e0 = tri.b - tri.a;
        let e1 = tri.c - tri.a;
        let n = e0.cross(&e1).normalize();

        for (v_i, v) in [&tri.a, &tri.b, &tri.c].iter().enumerate() {
            let pt = [v.x, v.y, v.z];
            let pt_i = tri_i * 3 + v_i;

            if let Some(ei) = get_equivalent(&kdtree, pt) {
                vert_normals[ei] += n;
            } else {
                vert_normals[pt_i] = n;
                kdtree.add(pt, pt_i).unwrap();
            }
        }
    }

    let mut triangle_normals: Vec<TriangleNormals> =
        unsafe { vec![std::mem::uninitialized(); mesh.len()] };

    let get_vn = |v: &Point3| {
        let pt = [v.x, v.y, v.z];
        vert_normals[get_equivalent(&kdtree, pt).expect("fetching calculated vertex normal")]
            .normalize()
    };

    for (tri_i, tri) in mesh.iter().enumerate() {
        triangle_normals[tri_i].a = get_vn(&tri.a);
        triangle_normals[tri_i].b = get_vn(&tri.b);
        triangle_normals[tri_i].c = get_vn(&tri.c);
    }

    triangle_normals
}

#[snoozy]
pub fn make_raster_mesh(
    ctx: &mut Context,
    mesh: &SnoozyRef<Vec<Triangle>>,
) -> Result<RasterGpuMesh> {
    let mesh = ctx.get(mesh)?;
    let mut verts: Vec<RasterGpuVertex> = Vec::with_capacity(mesh.len() * 3);
    let normals = calculate_vertex_normals(&*mesh);

    for (tri_i, tri) in mesh.iter().enumerate() {
        //let e0 = tri.b - tri.a;
        //let e1 = tri.c - tri.a;
        //let n = e0.cross(&e1).normalize();
        let n: &TriangleNormals = &normals[tri_i];

        for (v, n) in &[(&tri.a, &n.a), (&tri.b, &n.b), (&tri.c, &n.c)] {
            verts.push(RasterGpuVertex {
                pos: [v.x, v.y, v.z],
                normal: pack_unit_direction_11_10_11(n.x, n.y, n.z),
            });
        }
    }

    Ok(RasterGpuMesh {
        verts,
        index_count: (mesh.len() * 3) as u32,
    })
}

#[snoozy]
pub fn upload_raster_mesh(
    ctx: &mut Context,
    mesh: &SnoozyRef<RasterGpuMesh>,
) -> Result<ShaderUniformBundle> {
    let mesh = ctx.get(mesh)?;

    let verts = ArcView::new(&mesh, |m| &m.verts);

    Ok(shader_uniforms!(
        "mesh_vertex_buf": upload_array_buffer(verts),
        "mesh_index_count": mesh.index_count as u32
    ))
}
