use std::collections::HashMap;
use std::default::Default;
use std::mem::replace;
use std::sync::Mutex;

pub fn profile<F: FnOnce()>(name: &str, f: F) {
    GPU_PROFILER.lock().unwrap().profile(name, f);
}

pub fn end_frame() {
    let mut prof = GPU_PROFILER.lock().unwrap();
    prof.try_finish_queries();
}

pub fn with_stats<F: FnOnce(&GpuProfilerStats)>(f: F) {
    f(&GPU_PROFILER.lock().unwrap().stats);
}

// TODO: currently merges multiple invocations in a frame into a single bucket, and averages it
// should instead report the count per frame along with correct per-hit timing
#[derive(Debug, Default)]
pub struct GpuProfilerScope {
    pub hits: [u64; 16], // nanoseconds
    pub write_head: u32,
}

impl GpuProfilerScope {
    pub fn average_duration_millis(&self) -> f64 {
        let count = (self.write_head.min(self.hits.len() as u32) as f64).max(1.0);
        self.hits.iter().sum::<u64>() as f64 / count / 1_000_000.0
    }
}

#[derive(Default, Debug)]
pub struct GpuProfilerStats {
    pub scopes: HashMap<String, GpuProfilerScope>,
    pub order: Vec<String>,
}

struct ActiveQuery {
    handle: u32,
    name: String,
}

impl ActiveQuery {
    fn try_get_duration_nanos(&self) -> Option<u64> {
        let mut available: i32 = 0;
        unsafe {
            gl::GetQueryObjectiv(self.handle, gl::QUERY_RESULT_AVAILABLE, &mut available);
        }

        if available != 0 {
            let mut nanos = 0u64;
            unsafe {
                gl::GetQueryObjectui64v(self.handle, gl::QUERY_RESULT, &mut nanos);
            }

            Some(nanos)
        } else {
            None
        }
    }
}

impl GpuProfilerStats {
    fn report_duration_nanos(&mut self, name: String, duration: u64) {
        let mut entry = self.scopes.entry(name).or_default();
        entry.hits[entry.write_head as usize % entry.hits.len()] = duration;
        entry.write_head += 1;
    }
}

struct GpuProfiler {
    active_queries: Vec<ActiveQuery>,
    inactive_queries: Vec<u32>,
    frame_query_names: Vec<String>,
    stats: GpuProfilerStats,
}

impl GpuProfiler {
    pub fn new() -> Self {
        Self {
            active_queries: Vec::new(),
            inactive_queries: Vec::new(),
            frame_query_names: Vec::new(),
            stats: Default::default(),
        }
    }

    fn new_query_handle(&mut self) -> u32 {
        if let Some(h) = self.inactive_queries.pop() {
            h
        } else {
            let mut h = 0u32;
            unsafe {
                gl::GenQueries(1, &mut h);
            }
            h
        }
    }

    fn try_finish_queries(&mut self) {
        let finished_queries: Vec<(usize, (String, u64))> = self
            .active_queries
            .iter_mut()
            .enumerate()
            .filter_map(|(i, q)| {
                if let Some(duration) = q.try_get_duration_nanos() {
                    Some((i, (replace(&mut q.name, String::new()), duration)))
                } else {
                    None
                }
            })
            .collect();

        self.stats.order.clear();
        self.stats.order.extend(self.frame_query_names.drain(..));

        // Remove the finished queries from the active list
        for (i, _) in finished_queries.iter().rev() {
            self.inactive_queries.push(self.active_queries[*i].handle);
            self.active_queries.swap_remove(*i);
        }

        for (_, (name, duration)) in finished_queries.into_iter() {
            self.stats.report_duration_nanos(name, duration);
        }
    }

    fn profile<F: FnOnce()>(&mut self, name: &str, f: F) {
        self.frame_query_names.push(name.to_string());

        let handle = self.new_query_handle();
        unsafe {
            gl::BeginQuery(gl::TIME_ELAPSED, handle);
        }

        f();

        unsafe {
            gl::EndQuery(gl::TIME_ELAPSED);
        }
        self.active_queries.push(ActiveQuery {
            handle,
            name: name.to_string(),
        });
    }
}

lazy_static! {
    static ref GPU_PROFILER: Mutex<GpuProfiler> = { Mutex::new(GpuProfiler::new()) };
}
