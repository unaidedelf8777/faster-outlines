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
use anyhow::Result;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

const SPIECE_UNDERLINE: char = '\u{2581}';

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenVocabulary {
    pub tokens: Vec<(String, Vec<u32>)>,
    pub eos_token_id: u32,
}

impl TokenVocabulary {
    pub fn default() -> Self {
        Self {
            tokens: Vec::new(),
            eos_token_id: 0,
        }
    }
    pub fn from_hashmap(vocab_map: FxHashMap<String, Vec<u32>>, eos_token_id: u32) -> Self {
        let tokens: Vec<(String, Vec<u32>)> = vocab_map.into_iter().collect();
        TokenVocabulary {
            tokens,
            eos_token_id,
        }
    }

    pub fn from_raw_vocab(
        raw_vocab: FxHashMap<String, u32>,
        eos_token_id: u32,
        special_tokens: Option<FxHashSet<String>>,
    ) -> Result<Self> {
        if raw_vocab.is_empty() {
            bail!("Empty vocabulary provided");
        }

        let mut processed_vocab: FxHashMap<String, Vec<u32>> = FxHashMap::default();

        for (token, token_id) in raw_vocab {
            if let Some(ref special) = special_tokens {
                if special.contains(&token) {
                    continue;
                }
            }

            let processed_token = preprocess_token(&token)?;
            processed_vocab
                .entry(processed_token)
                .or_insert_with(Vec::new)
                .push(token_id);
        }

        Ok(TokenVocabulary {
            tokens: processed_vocab.into_iter().collect(),
            eos_token_id,
        })
    }

    /// Merges two TokenVocabulary instances, preserving EOS token from self.
    pub fn merge(self, other: TokenVocabulary) -> Self {
        let mut combined: FxHashMap<String, Vec<u32>> = FxHashMap::default();

        for (token, ids) in self.tokens {
            combined.insert(token, ids);
        }

        for (token, ids) in other.tokens {
            combined
                .entry(token)
                .and_modify(|existing| existing.extend(ids.iter()))
                .or_insert(ids);
        }

        TokenVocabulary {
            tokens: combined.into_iter().collect(),
            eos_token_id: self.eos_token_id,
        }
    }

    pub fn add_token(&mut self, token: String, values: Vec<u32>) {
        self.tokens.push((token, values));
    }

    pub fn remove_token(&mut self, token: &str) -> Option<Vec<u32>> {
        if let Some(pos) = self.tokens.iter().position(|(t, _)| t == token) {
            let (_, values) = self.tokens.remove(pos);
            return Some(values);
        }
        None
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &(String, Vec<u32>)> {
        self.tokens.iter()
    }
}

impl<'a> IntoIterator for &'a TokenVocabulary {
    type Item = (&'a String, &'a Vec<u32>);
    type IntoIter = std::iter::Map<
        std::slice::Iter<'a, (String, Vec<u32>)>,
        fn(&'a (String, Vec<u32>)) -> (&'a String, &'a Vec<u32>),
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.tokens.iter().map(|(s, v)| (&s, v))
    }
}

fn preprocess_token(token: &str) -> Result<String> {
    if token.is_empty() {
        bail!("Empty token provided");
    }

    let first_char = token.chars().next().unwrap();

    Ok(if first_char == SPIECE_UNDERLINE || token == "<0x20>" {
        format!(" {}", token)
    } else {
        token.to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_vocabulary_construction() {
        let mut vocab = FxHashMap::default();
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);

        let token_vocab = TokenVocabulary::from_raw_vocab(vocab, 0, None).unwrap();

        assert_eq!(token_vocab.len(), 2);
        assert_eq!(token_vocab.eos_token_id, 0);
    }

    #[test]
    fn test_special_token_handling() {
        let mut vocab = FxHashMap::default();
        vocab.insert(format!("{}word", SPIECE_UNDERLINE), 1);
        vocab.insert("<0x20>".to_string(), 2);

        let token_vocab = TokenVocabulary::from_raw_vocab(vocab, 0, None);

        let has_space_prefix = token_vocab.unwrap().iter().any(|(t, _)| t.starts_with(' '));

        assert!(has_space_prefix);
    }

    #[test]
    fn test_vocabulary_merge() {
        let mut vocab1 = FxHashMap::default();
        vocab1.insert("hello".to_string(), 1);

        let mut vocab2 = FxHashMap::default();
        vocab2.insert("world".to_string(), 2);

        let token_vocab1 = TokenVocabulary::from_raw_vocab(vocab1, 0, None);

        let token_vocab2 = TokenVocabulary::from_raw_vocab(vocab2, 1, None);

        let merged = token_vocab1.expect("").merge(token_vocab2.expect(""));

        assert_eq!(merged.len(), 2);
        assert_eq!(merged.eos_token_id, 0);
    }

    #[test]
    fn test_empty_vocabulary() {
        let vocab = FxHashMap::default();

        assert!(TokenVocabulary::from_raw_vocab(vocab, 0, None).is_err());
    }
}
