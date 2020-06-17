#![allow(dead_code)]
#![allow(unused_variables)]

use std::collections::HashMap;
use std::default::Default;
use std::sync::Mutex;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GpuProfilerQueryId(u64);

impl Default for GpuProfilerQueryId {
    fn default() -> Self {
        Self(std::u64::MAX)
    }
}

pub fn create_gpu_query(name: &str) -> GpuProfilerQueryId {
    GPU_PROFILER.lock().unwrap().create_gpu_query(name)
}

pub fn report_durations_ticks(
    ns_per_tick: f32,
    durations: impl Iterator<Item = (GpuProfilerQueryId, u64)>,
) {
    let mut prof = GPU_PROFILER.lock().unwrap();
    prof.report_durations_ticks(ns_per_tick, durations);
}

pub fn forget_queries(queries: impl Iterator<Item = GpuProfilerQueryId>) {
    let mut prof = GPU_PROFILER.lock().unwrap();
    prof.forget_queries(queries);
}

pub fn end_frame() {
    let mut prof = GPU_PROFILER.lock().unwrap();
    prof.end_frame();
}

pub fn with_stats<F: FnOnce(&GpuProfilerStats)>(f: F) {
    f(&GPU_PROFILER.lock().unwrap().stats);
}

pub fn get_stats() -> GpuProfilerStats {
    GPU_PROFILER.lock().unwrap().stats.clone()
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct GpuProfilerScopeId(String);

impl From<String> for GpuProfilerScopeId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

// TODO: currently merges multiple invocations in a frame into a single bucket, and averages it
// should instead report the count per frame along with correct per-hit timing
#[derive(Debug, Clone)]
pub struct GpuProfilerScope {
    pub name: String,
    pub hits: Vec<u64>, // nanoseconds
    pub write_head: u32,
}

impl GpuProfilerScope {
    fn with_name(name: String) -> GpuProfilerScope {
        GpuProfilerScope {
            hits: vec![0u64; 64],
            write_head: 0,
            name,
        }
    }
}

impl GpuProfilerScope {
    pub fn hit_count(&self) -> u32 {
        self.write_head.min(self.hits.len() as u32)
    }

    pub fn average_duration_millis(&self) -> f64 {
        let count = (self.hit_count() as f64).max(1.0);
        self.hits.iter().sum::<u64>() as f64 / count / 1_000_000.0
    }
}

#[derive(Default, Debug, Clone)]
pub struct GpuProfilerStats {
    pub scopes: HashMap<GpuProfilerScopeId, GpuProfilerScope>,
    pub order: Vec<GpuProfilerQueryId>,
}

struct ActiveQuery {
    id: GpuProfilerQueryId,
    name: String,
}

impl GpuProfilerStats {
    fn report_duration_nanos(&mut self, query_id: GpuProfilerQueryId, duration: u64, name: String) {
        let scope_id = GpuProfilerScopeId::from(name.clone());
        let mut entry = self
            .scopes
            .entry(scope_id)
            .or_insert_with(|| GpuProfilerScope::with_name(name));

        let len = entry.hits.len();
        entry.hits[entry.write_head as usize % len] = duration;
        entry.write_head += 1;
    }
}

struct GpuProfiler {
    active_queries: HashMap<GpuProfilerQueryId, ActiveQuery>,
    frame_query_ids: Vec<GpuProfilerQueryId>,
    next_query_id: u64,
    stats: GpuProfilerStats,
}

impl GpuProfiler {
    pub fn new() -> Self {
        Self {
            active_queries: Default::default(),
            frame_query_ids: Default::default(),
            next_query_id: 0,
            stats: Default::default(),
        }
    }

    fn report_durations_ticks(
        &mut self,
        ns_per_tick: f32,
        durations: impl Iterator<Item = (GpuProfilerQueryId, u64)>,
    ) {
        for (query_id, duration_ticks) in durations {
            // Remove the finished queries from the active list
            let q = self.active_queries.remove(&query_id).unwrap();
            let duration = (duration_ticks as f64 * ns_per_tick as f64) as u64;
            self.stats.report_duration_nanos(query_id, duration, q.name);
        }
    }

    fn forget_queries(&mut self, queries: impl Iterator<Item = GpuProfilerQueryId>) {
        for query_id in queries {
            let q = self.active_queries.remove(&query_id).unwrap();
        }
    }

    fn end_frame(&mut self) {
        self.stats.order.clear();
        self.stats.order.extend(self.frame_query_ids.drain(..));
    }

    fn create_gpu_query(&mut self, name: &str) -> GpuProfilerQueryId {
        let id = GpuProfilerQueryId(self.next_query_id);
        self.next_query_id += 1;
        self.frame_query_ids.push(id);
        self.active_queries.insert(
            id,
            ActiveQuery {
                id,
                name: name.to_string(),
            },
        );
        assert!(self.active_queries.len() < 8192);
        id
    }
}

lazy_static! {
    static ref GPU_PROFILER: Mutex<GpuProfiler> = Mutex::new(GpuProfiler::new());
}
