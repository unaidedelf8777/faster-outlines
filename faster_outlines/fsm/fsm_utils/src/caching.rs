use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use lru::LruCache;
use rustc_hash::FxHashMap;

use crate::types::TokenVocabulary;
use crate::environment::{FSM_CACHE_SIZE, DISABLE_CACHE};


/// Since iterating threw the entire vocab and getting a hash for it would be too costly,
/// we do the following hash function:
///     1. Get the first 1k tokens of vocab and hash them.
///     2. Hash the length of tokenizer.
///     3. Hash both length and hash of first 1k tokens together to get a combined hash.
/// 
/// This takes only 1-2ms, where hashing the whole vocab of 128k tokens can take up to 128ms ( 0.128 sec )
pub fn hash_token_vocabulary(vocabulary: &TokenVocabulary, pattern: &str) -> u64 {
    let mut hasher = DefaultHasher::new();


    vocabulary.len().hash(&mut hasher);


    pattern.hash(&mut hasher);


    let mut entries: Vec<(&String, &Vec<u32>)> = vocabulary.iter().collect();

    if entries.len() > 1000 {
        entries.select_nth_unstable_by(999, |a, b| a.0.cmp(b.0));
        entries.truncate(1000);
        entries.sort_by(|a, b| a.0.cmp(b.0));
    } else {
        entries.sort_by(|a, b| a.0.cmp(b.0));
    }

    for (key, value) in entries {
        key.hash(&mut hasher);
        value.hash(&mut hasher);
    }

    hasher.finish()
}


pub(crate) struct CachedFSM {
    pub states_to_token_maps: Vec<FxHashMap<u32,u32>>,
    pub first_state: u32,
    pub finals: Vec<u32>
}

pub(crate) struct ModuleState {
    pub fsm_cache: Mutex<LruCache<u64, Arc<CachedFSM>>>,
}

pub(crate) static MODULE_STATE: Lazy<ModuleState> = Lazy::new(|| {
    ModuleState {
        fsm_cache: Mutex::new(LruCache::new(std::num::NonZeroUsize::new(*FSM_CACHE_SIZE).unwrap())),
    }
});

pub fn get_cached_fsm(hash: u64) -> Option<Arc<CachedFSM>> {
    if *DISABLE_CACHE {
        return None;
    }

    let mut cache = MODULE_STATE.fsm_cache.lock().unwrap();
    if let Some(cached_fsm) = cache.get(&hash) {
        Some(Arc::clone(cached_fsm))
    } else {
        None
    }
}
