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
use anyhow::{Result, bail, anyhow};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use regex::Regex;
use once_cell::sync::Lazy;

use crate::sp_decode::{UNICODE_TO_BYTES, convert_tokens_to_string};

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
        from_sentencepiece: Option<bool>
    ) -> Result<Self> {
        if raw_vocab.is_empty() {
            bail!("Empty vocabulary provided");
        }

        let mut processed_vocab: FxHashMap<String, Vec<u32>> = FxHashMap::default();

        for (mut token, token_id) in raw_vocab {
            if let Some(ref special) = special_tokens {
                if special.contains(&token) {
                    continue;
                }
            }

            if from_sentencepiece.unwrap_or(false) {
                token = convert_tokens_to_string(vec![token]);
            }

            match preprocess_token(&token) {
                Ok(processed_token) => {
                    processed_vocab
                        .entry(processed_token)
                        .or_insert_with(Vec::new)
                        .push(token_id);
                },
                Err(e) => {
                    return Err(anyhow!("Failed to process token '{}': {}", token, e));
                }
            }
        }

        Ok(TokenVocabulary {
            tokens: processed_vocab.into_iter().collect(),
            eos_token_id,
        })
    }

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
            Some(values)
        } else {
            None
        }
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

static LLAMA_BYTE_TOKEN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^<0x[0-9A-F]{2}>$").unwrap()
});

static REPLACEMENT_SEQ_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^▁�+\.$").unwrap()
});

fn byte_to_symbol(byte: u8) -> String {
    if byte >= 0x80 {
        format!("\x00{:02X}", byte)
    } else {
        (byte as char).to_string()
    }
}

fn preprocess_token(token: &str) -> Result<String> {
    if token.is_empty() {
        return Ok(token.to_string());
    }

    let processed_token = if token == "<0x20>" {
        format!(" {}", token)
    } else {
        token.to_string()
    };

    if processed_token.contains('\u{fffd}') && !REPLACEMENT_SEQ_RE.is_match(&processed_token) {
        if LLAMA_BYTE_TOKEN_RE.is_match(&processed_token) {
            match u8::from_str_radix(&processed_token[3..5], 16) {
                Ok(byte) => return Ok(byte_to_symbol(byte)),
                Err(_) => return Err(anyhow!("Invalid byte in token")),
            }
        } else {
            let mut bytes = Vec::new();
            for c in processed_token.chars() {
                match UNICODE_TO_BYTES.get(&c) {
                    Some(&byte) => bytes.push(byte),
                    None => {
                        // If character not found, return the original token
                        return Ok(processed_token);
                    }
                }
            }
            return Ok(bytes.into_iter().map(byte_to_symbol).collect());
        }

    }
    Ok(processed_token)
}