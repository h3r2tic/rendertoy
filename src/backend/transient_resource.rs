pub use std::any::Any;

use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Mutex;
use typemap::{ShareMap, TypeMap};

pub trait TransientResource: Clone {
    type Desc: TransientResourceDesc;
    type Allocation: TransientResourceAllocPayload;

    fn new(
        desc: Self::Desc,
        allocation: std::sync::Arc<TransientResourceAllocation<Self::Desc, Self::Allocation>>,
    ) -> Self;
    fn allocate_payload(desc: Self::Desc) -> Self::Allocation;
}

// --------------------------------------------------------
// Trait collections

pub trait TransientResourceDesc:
    'static + Eq + Send + Sync + PartialEq + Hash + Clone + Copy
{
}
impl<T> TransientResourceDesc for T where
    T: 'static + Eq + Send + Sync + PartialEq + Hash + Clone + Copy
{
}

pub trait TransientResourceAllocPayload: 'static + Send + Sync + Clone {}
impl<T> TransientResourceAllocPayload for T where T: 'static + Send + Sync + Clone {}

pub trait TransientAllocation: Send + Sync + Drop {}

impl<Desc, AllocPayload> TransientAllocation for TransientResourceAllocation<Desc, AllocPayload>
where
    Desc: TransientResourceDesc,
    AllocPayload: TransientResourceAllocPayload,
{
}

// Type-erased allocation for storage in resource structs
pub type SharedTransientAllocation = std::sync::Arc<dyn TransientAllocation>;

// --------------------------------------------------------

// Key via which resources are matched for reuse.
#[derive(Clone)]
pub struct TransientResourceKey<Desc: TransientResourceDesc>(pub Desc);

// A live allocation of a resouce which can be shared over multiple handles,
// or wait in a pool for later use.
#[derive(Clone)]
pub struct TransientResourceAllocation<Desc, AllocPayload>
where
    Desc: TransientResourceDesc,
    AllocPayload: TransientResourceAllocPayload,
{
    pub key: TransientResourceKey<Desc>,
    pub payload: AllocPayload,
}

// https://stackoverflow.com/a/45893270
// Use `fn(T)` as it avoids having to require that `T` implement
// `Send + Sync`.
struct Key<Desc, P>(PhantomData<fn(Desc, P)>);

impl<Desc, P> typemap::Key for Key<Desc, P>
where
    Desc: TransientResourceDesc,
    P: TransientResourceAllocPayload,
{
    type Value = HashMap<Desc, Vec<TransientResourceAllocation<Desc, P>>>;
}

lazy_static! {
    static ref TRANSIENT_RESOURCE_CACHE: Mutex<ShareMap> = { Mutex::new(TypeMap::custom()) };
}

pub fn create_transient<Res: TransientResource>(desc: Res::Desc) -> Res {
    let mut res_cache_lock = TRANSIENT_RESOURCE_CACHE.lock().unwrap();
    let res_cache = res_cache_lock
        .entry::<Key<Res::Desc, Res::Allocation>>()
        .or_insert_with(|| HashMap::new());

    let existing = res_cache.entry(desc).or_default();

    let alloc = if existing.is_empty() {
        TransientResourceAllocation {
            key: TransientResourceKey(desc),
            payload: Res::allocate_payload(desc),
        }
    } else {
        existing.pop().unwrap()
    };

    Res::new(desc, std::sync::Arc::new(alloc))
}

impl<Desc, P> Drop for TransientResourceAllocation<Desc, P>
where
    Desc: TransientResourceDesc,
    P: TransientResourceAllocPayload,
{
    fn drop(&mut self) {
        let mut res_cache_lock = TRANSIENT_RESOURCE_CACHE.lock().unwrap();
        let res_cache = res_cache_lock
            .entry::<Key<Desc, P>>()
            .or_insert_with(|| HashMap::new());

        res_cache.entry(self.key.0).or_default().push(self.clone());
    }
}
