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

use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use lru::LruCache;

use crate::types::{TokenVocabulary, VocabTrie, VocabTrieBuilder};

// LRU cache for vocab tries.
// Set to 10 since there is no way you have more than 10 models with different tokenizers.
const TRIE_CACHE_SIZE: usize = 10;

pub(crate) struct ModuleState {
    vocab_trie_cache: Mutex<LruCache<u64, Arc<VocabTrie>>>,
}

pub(crate) static MODULE_STATE: Lazy<ModuleState> = Lazy::new(|| {
    ModuleState {
        vocab_trie_cache: Mutex::new(LruCache::new(std::num::NonZeroUsize::new(TRIE_CACHE_SIZE).unwrap())),
    }
});

pub fn hash_token_vocabulary(vocabulary: &TokenVocabulary) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Collect entries into a vector
    let mut entries: Vec<(&String, &Vec<u32>)> = vocabulary.iter().collect();

    // Sort entries by key to ensure deterministic hash
    entries.sort_by(|a, b| a.0.cmp(b.0));

    // Hash each key-value pair
    for (key, value) in entries {
        key.hash(&mut hasher);
        value.hash(&mut hasher);
    }

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
