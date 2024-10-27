// Copyright 2024 Nathan Hoos
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::environment::{FSM_CACHE_SIZE, DISABLE_CACHE};
use serde::{Serialize, Deserialize};
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
use lru::LruCache;

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct CachedFSM {
    pub states_to_token_maps: Vec<FxHashMap<u32,u32>>,
    pub first_state: u32,
    pub finals: Vec<u32>,
    pub hash: u64
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

pub fn insert_fsm_to_cache(cached_fsm: CachedFSM, cache_key: u64) {
    let mut cache = MODULE_STATE.fsm_cache.lock().unwrap();
    cache.put(cache_key, Arc::new(cached_fsm));
}