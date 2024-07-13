
use std::collections::{BTreeMap, BTreeSet};
/// since pyo3 cant convert to BTreeMap and BTreeSet
use std::collections::{HashMap, HashSet};

use std::convert::TryFrom;

use pyo3::prelude::*;

/// `TokenVocabulary` is a type alias for a `BTreeMap` where the key is a `char` representing a token, and the value is a `Vec<u32>` containing token IDs. This structure is used to store and manage tokens for processing with finite state machines (FSMs), ensuring ordered access and efficient retrieval without hashing overhead.
pub type TokenVocabulary = BTreeMap<String, Vec<u32>>;

#[derive(Debug, Clone, FromPyObject)]
pub struct PyFSMInfo {
    #[pyo3(item("initial"))]
    initial: u32,
    #[pyo3(item("finals"))]
    finals: HashSet<u32>,
    #[pyo3(item("transitions"))]
    transitions: HashMap<(u32, u32), u32>,
    //#[pyo3(item("trans_key_to_states"))]
    //trans_key_to_states: HashMap<u32, Vec<u32>>,
    #[pyo3(item("alphabet_anything_value"))]
    alphabet_anything_value: u32,

    #[pyo3(item("alphabet_symbol_mapping"))]
    alphabet_symbol_mapping: HashMap<String, u32>,
}

/// For mapping PyFSMInfo to std FSMInfo object.

impl TryFrom<&PyFSMInfo> for FSMInfo {
    type Error = &'static str; // Simplify error handling for this example

    fn try_from(py_info: &PyFSMInfo) -> Result<Self, Self::Error> {
        // Direct assignment as the type is assumed to be correct
        let initial: u32 = py_info.initial;

        // Directly use the values assuming they are already u32
        let finals = py_info.finals.iter().copied().collect::<Vec<u32>>();
        
        // Transitions conversion assuming all parts are already u32
        let mut transitions = BTreeMap::new();
        for (&(from_state, input), &to_state) in &py_info.transitions {
            transitions.insert((from_state, input), to_state);
        }

        // Alphabet symbol mapping conversion assuming all values are already u32
        let alphabet_symbol_mapping = py_info
            .alphabet_symbol_mapping
            .iter()
            .map(|(symbol, &trans_key)| (symbol.clone(), trans_key))
            .collect::<BTreeMap<String, u32>>();

        // Extract states from transitions
        let mut states = BTreeSet::new();
        for ((from, _), to) in &transitions {
            states.insert(*from);
            states.insert(*to);
        }



        let alphabet_anything_value = py_info.alphabet_anything_value;

        Ok(FSMInfo {
            initial,
            finals,
            transitions,
            alphabet_symbol_mapping,
            alphabet_anything_value,
            states,
        })
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FSMInfo {
    /// Initial state of the FSM
    pub initial: u32,
    /// Final states of the FSM
    pub finals: Vec<u32>,
    /// The transitions map of the fsm.
    /// key is a tuple, (state, input), where the state is the current state, and input is the transition
    /// Value is the state we will end up at if we take the transition
    /// there can be multiple transitions for each state.
    pub transitions: BTreeMap<(u32, u32), u32>,

    /// The alphabet mapping.
    /// key is a String representing the input, value is its u32 identifier / transition key.
    pub alphabet_symbol_mapping: BTreeMap<String, u32>,

    pub alphabet_anything_value: u32,
    pub states: BTreeSet<u32>,
}

/// `VocabTrie` is a structure designed to efficiently index the vocabulary of a tokenizer. 
/// It facilitates the quick lookup of tokens based on their prefixes and manages relationships between tokens and their substrings, 
/// for operations like token scanning in FSMs.
///
/// - **`parent_children_map`**: Maps a token prefix ID to a vector of child token IDs, enabling quick exploration of possible token continuations.
/// - **`idx_to_token_str`**: Provides an index-based lookup from a token ID to the corresponding token string.
/// - **`token_str_to_idx`**: Maps a token string to its unique ID, facilitating fast conversions from strings to indices.
#[derive(Clone)]
pub struct VocabTrie {
    parent_children_map: Vec<Option<Vec<u32>>>,
    idx_to_token_str: Vec<TrieToken>,
    root_tokens: Vec<TrieToken>,
}

impl VocabTrie {
    #[inline(always)]
    pub fn find_children(&self, token_idx: u32) -> Option<Vec<&TrieToken>> {
        self.parent_children_map.get(token_idx as usize)
            .and_then(|children| children.as_ref())
            .map(|children| {
                children.iter()
                    .filter_map(|&child_id| self.get_token(child_id))
                    .collect()
            })
    }

    #[inline(always)]
    pub fn get_token(&self, index: u32) -> Option<&TrieToken> {
        self.idx_to_token_str.get(index as usize)
    }

    #[inline(always)]
    pub fn get_root_tokens(&self) -> &Vec<TrieToken> {
        &self.root_tokens
    }
}

#[derive(Clone, Debug)]
pub struct TrieToken {
    pub tok_id: usize,
    pub idx: u32,
    pub str_len: usize,
}

pub trait VocabTrieBuilder {
    fn to_vocab_trie(&self) -> VocabTrie;
}

impl VocabTrieBuilder for TokenVocabulary {
    fn to_vocab_trie(&self) -> VocabTrie {
        let max_token_id = self.values().flatten().max().map(|&id| id as usize + 1).unwrap_or(0);
        let mut parent_children_map = vec![None; max_token_id];
        let mut idx_to_token_str = Vec::new();
        let mut token_str_to_idx: BTreeMap<String, u32> = BTreeMap::new();
        let mut root_tokens = Vec::new();

        let mut token_id: u32 = 0;
        for (token, ids) in self.iter() {
            let trie_token = TrieToken {
                tok_id: ids[0] as usize,
                idx: token_id,
                str_len: token.chars().count(),
            };
            idx_to_token_str.push(trie_token.clone());
            token_str_to_idx.insert(token.clone(), token_id);

            let char_indices: Vec<usize> = token.char_indices().map(|(index, _)| index).collect();
            let mut is_root = true;
            for i in 0..char_indices.len() - 1 {
                let prefix = &token[..char_indices[i]];
                if self.contains_key(prefix) {
                    let prefix_id = *token_str_to_idx.get(prefix).unwrap();
                    parent_children_map[prefix_id as usize]
                        .get_or_insert_with(Vec::new)
                        .push(token_id);
                    is_root = false;
                }
            }
            if is_root {
                root_tokens.push(trie_token);
            }
            token_id += 1;
        }

        VocabTrie {
            parent_children_map,
            idx_to_token_str,
            root_tokens,
        }
    }
}
