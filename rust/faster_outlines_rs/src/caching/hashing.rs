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

use crate::vocab::TokenVocabulary;
use std::collections::hash_map::DefaultHasher;
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

    if vocabulary.len() > 100 {
        let partition_key = vocabulary.iter().map(|(k, _)| k).nth(99).unwrap();

        for (key, value) in vocabulary.iter().filter(|(k, _)| k <= &partition_key) {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }
    } else {
        for (key, value) in vocabulary.iter() {
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
