use super::*;

use bvh::{
    aabb::{Bounded, AABB},
    bounding_hierarchy::BHShape,
    bvh::{BVHNode, BVH},
};

impl Bounded for Triangle {
    fn aabb(&self) -> AABB {
        AABB::empty().grow(&self.a).grow(&self.b).grow(&self.c)
    }
}
impl BHShape for Triangle {}

#[derive(Clone, Copy)]
#[repr(C)]
struct GpuBvhNode {
    packed: [u32; 4],
}

fn pack_gpu_bvh_node(node: BvhNode) -> GpuBvhNode {
    let bmin = (
        half::f16::from_f32(node.bbox_min.x),
        half::f16::from_f32(node.bbox_min.y),
        half::f16::from_f32(node.bbox_min.z),
    );

    let box_extent_packed = {
        // The fp16 was rounded-down, so extent will be larger than for fp32
        let extent =
            node.bbox_max - Vector3::new(bmin.0.to_f32(), bmin.1.to_f32(), bmin.2.to_f32());

        rgb9e5::pack_rgb9e5(extent.x, extent.y, extent.z)
    };

    assert!(node.exit_idx < (1u32 << 24));
    assert!(node.prim_idx == std::u32::MAX || node.prim_idx < (1u32 << 24));

    GpuBvhNode {
        packed: [
            box_extent_packed,
            ((bmin.0.to_bits() as u32) << 16) | (bmin.1.to_bits() as u32),
            ((bmin.2.to_bits() as u32) << 16) | ((node.prim_idx >> 8) & 0xffff),
            ((node.prim_idx & 0xff) << 24) | node.exit_idx,
        ],
    }
}

struct BvhNode {
    bbox_min: Point3,
    exit_idx: u32,
    bbox_max: Point3,
    prim_idx: u32,
}

impl BvhNode {
    fn new_leaf(bbox_min: Point3, bbox_max: Point3, prim_idx: usize) -> Self {
        Self {
            bbox_min,
            exit_idx: 0,
            bbox_max,
            prim_idx: prim_idx as u32,
        }
    }

    fn new_interior(bbox_min: Point3, bbox_max: Point3) -> Self {
        Self {
            bbox_min,
            exit_idx: 0,
            bbox_max,
            prim_idx: std::u32::MAX,
        }
    }

    fn set_exit_idx(&mut self, idx: usize) {
        self.exit_idx = idx as u32;
    }

    fn get_exit_idx(&mut self) -> usize {
        self.exit_idx as usize
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct GpuTriangle {
    v: Point3,
    e0: Vector3,
    e1: Vector3,
}

assert_eq_size!(triangle_size_check; GpuTriangle, [u8; 9 * 4]);

fn convert_bvh<BoxOrderFn>(
    node: usize,
    nbox: &AABB,
    nodes: &[BVHNode],
    are_boxes_correctly_ordered: &BoxOrderFn,
    res: &mut Vec<BvhNode>,
) where
    BoxOrderFn: Fn(&AABB, &AABB) -> bool,
{
    let initial_node_count = res.len();
    let n = &nodes[node];

    let node_res_idx = if node != 0 {
        res.push(if let BVHNode::Node { .. } = n {
            BvhNode::new_interior(nbox.min, nbox.max)
        } else {
            BvhNode::new_leaf(
                nbox.min,
                nbox.max,
                n.shape_index().expect("bvh leaf shape index"),
            )
        });
        Some(initial_node_count)
    } else {
        None
    };

    if let BVHNode::Node { .. } = n {
        let boxes = [&n.child_l_aabb(), &n.child_r_aabb()];
        let indices = [n.child_l(), n.child_r()];

        let (first, second) = if are_boxes_correctly_ordered(boxes[0], boxes[1]) {
            (0, 1)
        } else {
            (1, 0)
        };

        convert_bvh(
            indices[first],
            &boxes[first],
            nodes,
            are_boxes_correctly_ordered,
            res,
        );
        convert_bvh(
            indices[second],
            &boxes[second],
            nodes,
            are_boxes_correctly_ordered,
            res,
        );
    }

    if let Some(node_res_idx) = node_res_idx {
        let index_after_subtree = res.len();
        res[node_res_idx].set_exit_idx(index_after_subtree);
    } else {
        // We are back at the root node. Go and change exit pointers to be relative,
        for (i, node) in res.iter_mut().enumerate().skip(initial_node_count) {
            let idx = node.get_exit_idx();
            node.set_exit_idx(idx - i);
        }
    }
}

snoozy! {
    fn build_gpu_bvh(ctx: &mut Context, mesh: &SnoozyRef<Vec<Triangle>>) -> Result<ShaderUniformBundle> {
        let mesh = ctx.get(mesh)?;

        let time0 = std::time::Instant::now();
        let bvh = BVH::build(&mesh);
        println!("BVH built in {:?}", time0.elapsed());

        let orderings = (
            |a: &AABB, b: &AABB| a.min.x + a.max.x < b.min.x + b.max.x,
            |a: &AABB, b: &AABB| a.min.x + a.max.x > b.min.x + b.max.x,
            |a: &AABB, b: &AABB| a.min.y + a.max.y < b.min.y + b.max.y,
            |a: &AABB, b: &AABB| a.min.y + a.max.y > b.min.y + b.max.y,
            |a: &AABB, b: &AABB| a.min.z + a.max.z < b.min.z + b.max.z,
            |a: &AABB, b: &AABB| a.min.z + a.max.z > b.min.z + b.max.z,
        );

        let time0 = std::time::Instant::now();

        let mut bvh_nodes: Vec<BvhNode> = Vec::with_capacity(bvh.nodes.len() * 6);

        macro_rules! ordered_flatten_bvh {
            ($order: expr) => {{
                convert_bvh(
                    0,
                    &AABB::default(),
                    bvh.nodes.as_slice(),
                    &$order,
                    &mut bvh_nodes,
                );
            }};
        }

        ordered_flatten_bvh!(orderings.0);
        ordered_flatten_bvh!(orderings.1);
        ordered_flatten_bvh!(orderings.2);
        ordered_flatten_bvh!(orderings.3);
        ordered_flatten_bvh!(orderings.4);
        ordered_flatten_bvh!(orderings.5);

        println!("BVH flattened in {:?}", time0.elapsed());
        let time0 = std::time::Instant::now();

        let gpu_bvh_nodes: Vec<_> = bvh_nodes.into_iter().map(pack_gpu_bvh_node).collect();

        let bvh_triangles = mesh
            .iter()
            .map(|t| GpuTriangle {
                v: t.a,
                e0: t.b - t.a,
                e1: t.c - t.a,
            })
            .collect::<Vec<_>>();

        println!("BVH encoded in {:?}", time0.elapsed());

        Ok(shader_uniforms!(
            "bvh_meta_buf": upload_buffer(to_byte_vec(vec![(gpu_bvh_nodes.len() / 6) as u32])),
            "bvh_nodes_buf": upload_buffer(to_byte_vec(gpu_bvh_nodes)),
            "bvh_triangles_buf": upload_buffer(to_byte_vec(bvh_triangles)),
        ))
    }
}
