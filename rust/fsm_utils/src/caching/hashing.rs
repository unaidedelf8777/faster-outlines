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

use std::collections::hash_map::DefaultHasher;
use crate::vocab::TokenVocabulary;
use std::hash::{Hash, Hasher};

// Since iterating threw the entire vocab and getting a hash for it would be too costly,
// we do the following hash function:
//     1. Get the first 100 tokens of vocab and hash them.
//     2. Hash the length of tokenizer.
//     3. Hash both length and hash of first 100 tokens together to get a combined hash.
// 
// This takes only nearly no time, where hashing the whole vocab of 128k tokens can take up to 128ms, which is way too long
pub fn hash_token_vocabulary(vocabulary: &TokenVocabulary) -> u64 {
    let mut hasher = DefaultHasher::new();
    vocabulary.len().hash(&mut hasher);

    if vocabulary.tokens.len() > 100 {
        let partition_key = vocabulary.tokens.iter()
            .map(|(k, _)| k)
            .nth(99)
            .unwrap();

        for (key, value) in vocabulary.tokens.iter()
            .filter(|(k, _)| k <= partition_key)
        {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }
    } else {
        for (key, value) in vocabulary.tokens.iter() {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }
    }

    hasher.finish()
}

pub fn get_fsm_cache_key(pattern: &str, vocabulary: &TokenVocabulary) -> u64 {
    let vocab_hash = hash_token_vocabulary(vocabulary);
    let mut hasher = DefaultHasher::new();

    pattern.hash(&mut hasher);
    vocab_hash.hash(&mut hasher);

    hasher.finish()
}

#[test]
fn test_hash_token_vocabulary() {
   use rustc_hash::FxHashMap;

   let mut token_to_ids = FxHashMap::default();
   
   for i in (0..150).rev() {
       token_to_ids.insert(format!("{:03}", i), vec![i as u32]);
   }
   
   let eos_token_id = 42;
   let vocab = TokenVocabulary::from_hashmap(token_to_ids.clone(), eos_token_id);
   let hash1 = hash_token_vocabulary(&vocab);
   
   let mut token_to_ids2 = token_to_ids;
   token_to_ids2.insert("000".to_string(), vec![999]);
   let vocab2 = TokenVocabulary::from_hashmap(token_to_ids2, eos_token_id);
   let hash2 = hash_token_vocabulary(&vocab2);
   
   assert_ne!(hash1, hash2);
}