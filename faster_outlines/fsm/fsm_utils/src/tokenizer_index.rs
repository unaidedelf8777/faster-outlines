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

/// * Imports * ///
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

use crate::caching::get_or_create_vocab_trie;
use crate::lazy_index::StateNotifierMap;
use crate::types::{FSMInfo, ThreadSafeCell, TokenVocabulary, TrieToken, VocabTrie};
use crate::{lazy_index::LazyFSMIndex, types::PyFSMInfo};
use std::convert::TryFrom;

fn create_token_transition_table(
    fsm_info: &FSMInfo,
    vocabulary: &TokenVocabulary,
    max_token_id: usize,
) -> Vec<Option<Vec<u32>>> {
    let mut table: Vec<Option<Vec<u32>>> = vec![None; max_token_id];

    for (token, tok_ids) in vocabulary.iter() {
        let chars: Vec<char> = token.chars().collect(); // Collect characters into a vector
        let char_count = chars.len(); // Get the number of characters
        let mut token_transition_keys: Vec<u32> = Vec::with_capacity(char_count); // Preallocate memory for the vector

        for symbol in &chars {
            // Iterate over the vector of characters
            token_transition_keys.push(
                *fsm_info
                    .alphabet_symbol_mapping
                    .get(&symbol.to_string())
                    .unwrap_or(&fsm_info.alphabet_anything_value),
            )
        }

        for t in tok_ids {
            table[*t as usize] = Some(token_transition_keys.clone());
        }
    }

    table
}

/// We only need the string for the trie construction. thus this vector suffices.
/// it acts as a map of a token string to its equivalent token ids.
/// we use the first token id from the vocab as the key index.
#[inline(always)]
fn create_vocab_to_tokenid_vector(
    vocab: &TokenVocabulary,
    max_token_id: usize,
) -> Vec<Option<Vec<u32>>> {
    let mut result_vec: Vec<Option<Vec<u32>>> = vec![None; max_token_id];

    for (_, token_ids) in vocab {
        if let Some(first_id) = token_ids.first() {
            result_vec[*first_id as usize] = Some(token_ids.clone());
        }
    }

    result_vec
}

fn walk_fsm(
    fsm_info: &FSMInfo,
    start_state: u32,
    transitions_vec: &[u32],
    full_match: bool,
) -> Vec<u32> {
    let mut state = start_state;
    let mut accepted_states = Vec::new();
    let mut last_final_idx = 0;

    for i in 0..transitions_vec.len() {
        let trans_key = transitions_vec[i];

        let new_state = fsm_info.transitions.get(&(state, trans_key));

        if let Some(&new_state) = new_state {
            state = new_state;
            if fsm_info.finals.contains(&state) {
                last_final_idx = i + 1; // Store the index of the last final state encountered
            }
            accepted_states.push(state);
        } else if !full_match && last_final_idx > 0 {
            return accepted_states.into_iter().take(last_final_idx).collect();
        } else {
            return Vec::new();
        }
    }

    if full_match && last_final_idx - 1 != transitions_vec.len() - 1 {
        Vec::new() // If full match is required and last final state is not at the end, return empty
    } else {
        accepted_states
    }
}

#[inline(always)]
fn state_scan_tokens(
    fsm_info: &FSMInfo,
    vocab_trie: &VocabTrie,
    vocab_vec: &Vec<Option<Vec<u32>>>,
    start_state: u32,
    transitions_table: &[Option<Vec<u32>>],
) -> FxHashSet<(u32, u32)> {
    // In this implementation, unlike outlines which iterates through every token in the tokenizer,
    // for each state, we use a Trie of the vocabulary, so that our searching is more efficient.
    // this way we are not wasting time on tokens which have prefixes which are not-useable / non-matching.
    // this is very helpful in terms of efficiency for characters such as `{`, for which there may only be a few
    // matching tokens in the tokenizer.
    let mut results = FxHashSet::default();
    // Initialize a local stack with a copy of the indices from root_tokens
    let mut stack: Vec<TrieToken> = vocab_trie.get_root_tokens().clone();

    // Process the tokens using the local stack
    while let Some(t) = stack.pop() {
        if let Some(token) = vocab_trie.get_token(t.idx) {
            let token_transitions = &transitions_table[token.tok_id];
            if let Some(transition_keys) = token_transitions {
                let state_seq = walk_fsm(fsm_info, start_state, transition_keys, false);

                if state_seq.len() == token.str_len {
                    if let Some(token_ids) = &vocab_vec[token.tok_id] {
                        let last_state = *state_seq.last().unwrap(); // Safe to unwrap because we check length == token.len()
                        for token_id in token_ids {
                            results.insert((*token_id, last_state));
                        }
                    }
                }

                // Always add successors to the stack
                if let Some(next_token_idxs) = vocab_trie.find_children(t.idx) {
                    stack.extend(next_token_idxs.iter().map(|&x| x.clone()));
                }
            }
        }
    }

    results
}

pub fn create_fsm_index_end_to_end_parallel(
    fsm_info: &FSMInfo,
    vocabulary: &TokenVocabulary,
    return_to: &Arc<Vec<ThreadSafeCell<FxHashMap<u32, u32>>>>,
    state_notifiers: &StateNotifierMap,
) {

    let max_token_id = vocabulary
        .values()
        .flat_map(|ids| ids.iter())
        .max()
        .map(|&id| id as usize + 1)
        .unwrap_or(0);

    let transitions_table = create_token_transition_table(fsm_info, vocabulary, max_token_id);
    let vocab_vec = create_vocab_to_tokenid_vector(vocabulary, max_token_id);
    let vocab_trie = get_or_create_vocab_trie(vocabulary);

    fsm_info.states.par_iter().for_each(|&start_state| {
        // Compute token_ids_end_states
        let token_ids_end_states = state_scan_tokens(
            fsm_info,
            &vocab_trie,
            &vocab_vec,
            start_state,
            &transitions_table,
        );

        // Unsafe mutable access without locking
        unsafe {
            let map = return_to[start_state as usize].get();
            for &(token_id, end_state) in &token_ids_end_states {
                map.insert(token_id, end_state);
            }
        }

        // Notify that the state is done
        let notifier = Arc::clone(&state_notifiers[start_state as usize]);
        let (done_lock, condvar) = &*notifier;
        let mut done = done_lock.lock().unwrap();
        *done = true;
        condvar.notify_all();
    });
}

/// Create an FSM state-to-vocabulary map/index through end-to-end token parsing. ///
#[pyfunction(name = "create_fsm_index_end_to_end")]
#[pyo3(text_signature = "(fsm_info, vocabulary, /)")]
pub fn create_fsm_index_end_to_end_py(
    py: Python<'_>,
    py_fsm_info: PyFSMInfo,
    vocabulary: TokenVocabulary,
    eos_token_id: u32,
) -> LazyFSMIndex {
    py.allow_threads(move || {
        let fsm_info = FSMInfo::try_from(&py_fsm_info).unwrap();
        LazyFSMIndex::new(fsm_info, vocabulary, eos_token_id)
    })
}
