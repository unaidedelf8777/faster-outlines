/* The MIT License (MIT)
* Copyright (c) 2024 Nathan Hoos
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

use crate::{
    environment::{DISABLE_CACHE, FSM_CACHE_SIZE},
    types::ThreadSafeCell,
};
use lru::LruCache;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub(crate) struct CachedFSM {
    pub states_to_token_maps: Arc<Vec<ThreadSafeCell<FxHashMap<u32, u32>>>>,
    pub first_state: u32,
    pub finals: Vec<u32>,
    pub hash: u64,
}

pub(crate) struct ModuleState {
    pub fsm_cache: Mutex<LruCache<u64, Arc<CachedFSM>>>,
}

pub(crate) static MODULE_STATE: Lazy<ModuleState> = Lazy::new(|| ModuleState {
    fsm_cache: Mutex::new(LruCache::new(
        std::num::NonZeroUsize::new(*FSM_CACHE_SIZE).unwrap(),
    )),
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
