use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use lru::LruCache;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod lazy_index;
mod tokenizer_index;
mod types;

use tokenizer_index::create_fsm_index_end_to_end_py;
use crate::lazy_index::{LazyFSMIndex, Write, Generate};
use crate::types::{TokenVocabulary, VocabTrie, VocabTrieBuilder};

// LRU cache size
const CACHE_SIZE: usize = 100;

struct ModuleState {
    vocab_trie_cache: Mutex<LruCache<u64, Arc<VocabTrie>>>,
}

static MODULE_STATE: Lazy<ModuleState> = Lazy::new(|| {
    ModuleState {
        vocab_trie_cache: Mutex::new(LruCache::new(std::num::NonZeroUsize::new(CACHE_SIZE).unwrap())),
    }
});

pub fn hash_token_vocabulary(vocabulary: &TokenVocabulary) -> u64 {
    let mut hasher = DefaultHasher::new();
    vocabulary.hash(&mut hasher);
    hasher.finish()
}

pub fn get_or_create_vocab_trie(vocabulary: &TokenVocabulary) -> Arc<VocabTrie> {
    let hash = hash_token_vocabulary(vocabulary);
    
    let mut cache = MODULE_STATE.vocab_trie_cache.lock().unwrap();
    
    if let Some(trie) = cache.get(&hash) {
        Arc::clone(trie)
    } else {
        let new_trie = Arc::new(vocabulary.to_vocab_trie());
        cache.put(hash, Arc::clone(&new_trie));
        new_trie
    }
}

#[pymodule]
fn fsm_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Ensure MODULE_STATE is initialized
    Lazy::force(&MODULE_STATE);

    m.add_function(wrap_pyfunction!(create_fsm_index_end_to_end_py, m)?)?;
    m.add_class::<LazyFSMIndex>()?;
    m.add_class::<Write>()?;
    m.add_class::<Generate>()?;
    Ok(())
}