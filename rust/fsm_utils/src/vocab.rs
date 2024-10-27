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
use rustc_hash::FxHashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenVocabulary {
    pub tokens: Vec<(String, Vec<u32>)>,
    pub eos_token_id: u32
}

impl TokenVocabulary {

    pub fn default() -> Self {
        Self {
            tokens: Vec::new(),
            eos_token_id: 0
        }
    }
    pub fn from_hashmap(vocab_map: FxHashMap<String, Vec<u32>>, eos_token_id: u32) -> Self {
        let tokens: Vec<(String, Vec<u32>)> = vocab_map.into_iter().collect();
        TokenVocabulary { tokens, eos_token_id }
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
        fn(&'a (String, Vec<u32>)) -> (&'a String, &'a Vec<u32>)
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.tokens.iter().map(|(s, v)| (&s, v))
    }
}