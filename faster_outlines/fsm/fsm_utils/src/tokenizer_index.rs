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

/// This module provides utility functions for tokenizing input strings using a finite state machine (FSM).
///
/// The `walk_fsm` function performs a FSM walk on an input string, starting from a given state, and returns a vector of accepted states.
///
/// The `state_scan_tokens` function scans a token vocabulary using the FSM and returns a set of token IDs and their corresponding end states.
///
/// The `create_fsm_index_end_to_end_parallel` function creates an FSM state-to-vocabulary map/index through end-to-end token parsing in parallel.
///
/// The `trim_vocabulary` function trims the token vocabulary by filtering out tokens that contain characters not present in the FSM's alphabet.
///
/// The `create_fsm_index_end_to_end_py` function is a Python interface for creating an FSM state-to-vocabulary map/index through end-to-end token parsing.
/// It takes a pattern string, a token vocabulary, and an end-of-sequence token ID as input, and returns a LazyFSMIndex object.

/// * Imports * ///
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Condvar, Mutex};

use crate::get_or_create_vocab_trie;
use crate::lazy_index::StateNotifierMap;
use crate::types::{FSMInfo, TokenVocabulary, TrieToken, VocabTrie};
use crate::{lazy_index::LazyFSMIndex, types::PyFSMInfo};
use dashmap::DashMap;
use std::convert::TryFrom;

fn create_token_transition_table(
    fsm_info: &Arc<FSMInfo>,
    vocabulary: &Arc<TokenVocabulary>,
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
fn create_vocab_to_tokenid_vector(vocab: &Arc<TokenVocabulary>, max_token_id: usize) -> Vec<Option<Vec<u32>>> {
    let mut result_vec: Vec<Option<Vec<u32>>> = vec![None; max_token_id];
    
    for (_, token_ids) in vocab.as_ref() {
        if let Some(first_id) = token_ids.get(0) {
            result_vec[*first_id as usize] = Some(token_ids.clone());
        }
    }
    
    result_vec
}

/// This function performs a walk through a finite state machine (FSM) based on the provided input string starting from a specified state.
///
/// **Arguments:**
/// - `fsm_info`: Reference to `FSMInfo` which holds the details about the FSM, including transitions and final states.
/// - `input_string`: The string input to be tokenized or processed through the FSM.
/// - `start_state`: The initial state from which the FSM walk begins.
/// - `full_match`: A boolean that determines if the function should return results only when the entire input string is matched to a final state.

/// **Returns:**
/// - `Vec<u32>`: A vector of accepted states after processing the input string through the FSM.
/// This vector includes states that are reached which form a part of the final states of the FSM, depending on the `full_match` requirement.
///
/// **Description:**
/// The function iterates over the input string, trying to match larger substrings first to accommodate multi-character transitions in the FSM.
/// If a substring matches and leads to a state that is a final state, it records this position. Depending on the `full_match` flag,
/// it may return early or continue to process until all substrings are attempted. The function is sensitive to the ordering of characters
/// and transitions, ensuring that the longest possible matches are considered first.
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

/// This function scans a set of tokens against an FSM to determine the resulting states from a given start state.
///
/// **Arguments:**
/// - `fsm_info`: Reference to `FSMInfo` containing FSM transition rules and other metadata.
/// - `vocabulary`: Reference to `TokenVocabulary`, a collection of tokens that are to be tested against the FSM.
/// - `start_state`: The initial state from which token scanning begins in the FSM.
///
/// **Returns:**
/// - `BTreeSet<(u32, u32)>`: A set of tuples where each tuple consists of a token ID and the corresponding end state in the FSM after processing the token.
///
/// **Description:**
/// The function iterates over each token in the vocabulary and applies `walk_fsm` to determine how far the token can be processed within the FSM starting from the `start_state`.
/// If a token can be fully processed (i.e., the length of the state sequence returned by `walk_fsm` matches the token length), the end state and token ID are recorded.
/// The results are unique due to the nature of `BTreeSet`, ensuring no duplicate entries for tokens leading to the same end state.
#[inline(always)]
fn state_scan_tokens(
    fsm_info: &FSMInfo,
    vocab_trie: &VocabTrie,
    vocab_vec: &Vec<Option<Vec<u32>>>,
    start_state: u32,
    transitions_table: &[Option<Vec<u32>>],
) -> BTreeSet<(u32, u32)> {
    let mut results = BTreeSet::new();
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

/// Creates a mapping of FSM states to vocabulary tokens in parallel, facilitating quick construction of state-to-token maps for large vocabularies.
///
/// **Arguments:**
/// - `fsm_info`: An `Arc<FSMInfo>` encapsulating the FSM's transition rules and final states.
/// - `vocabulary`: An `Arc<TokenVocabulary>` representing the set of tokens to be used.
/// - `return_to`: An `Arc<Mutex<BTreeMap<u32, BTreeMap<u32, u32>>>>` where the resulting state-to-token map is stored.
/// - `state_notifiers`: An `Arc<Mutex<BTreeMap<u32, Arc<(Mutex<bool>, Condvar)>>>>` used to notify other threads about the completion of state computations.
///
/// **Returns:**
/// - `None`: The function returns nothing, but it populates `return_to` with the computed state-to-token maps and notifies other processes of completion through `state_notifiers`.
///
/// **Description:**
/// The function processes each state in the FSM in parallel, applying `state_scan_tokens` to build a map from each state to possible tokens and their resultant states.
/// It fills `return_to` with these mappings and uses `state_notifiers` to signal the completion of the computation for each state,
///  enabling efficient multi-threaded computation and synchronization.
pub fn create_fsm_index_end_to_end_parallel(
    fsm_info: &Arc<FSMInfo>,
    vocabulary: &Arc<TokenVocabulary>,
    return_to: &Arc<DashMap<u32, BTreeMap<u32, u32>>>,
    state_notifiers: &StateNotifierMap,
) {
    let max_token_id = vocabulary
        .values()
        .flat_map(|ids| ids.iter())
        .max()
        .map(|&id| id as usize + 1)
        .unwrap_or(0);
    let transitions_table = Arc::new(Mutex::new(None));
    let vocab_vec = Arc::new(Mutex::new(None));
    let vocab_trie = Arc::new(Mutex::new(None));

    // Compute the variables in parallel
    (0..3).into_par_iter().for_each(|i| {
        match i {
            0 => {
                let table = create_token_transition_table(fsm_info, vocabulary, max_token_id);
                *transitions_table.lock().unwrap() = Some(table);
            },
            1 => {
                let vec = create_vocab_to_tokenid_vector(vocabulary, max_token_id);
                *vocab_vec.lock().unwrap() = Some(vec);
            },
            2 => {
                let trie = get_or_create_vocab_trie(vocabulary);
                *vocab_trie.lock().unwrap() = Some(trie);
            },
            _ => unreachable!(),
        }
    });

    // Ensure all data structures are initialized
    let transitions_table = transitions_table.lock().unwrap().as_ref().unwrap().clone();
    let vocab_vec = vocab_vec.lock().unwrap().as_ref().unwrap().clone();
    let vocab_trie = vocab_trie.lock().unwrap().as_ref().unwrap().clone();

    fsm_info.states.par_iter().for_each(|&start_state| {
        let token_ids_end_states = state_scan_tokens(
            fsm_info,
            &vocab_trie,
            &vocab_vec,
            start_state,
            &transitions_table,
        );

        let mut map = BTreeMap::new();
        for &(token_id, end_state) in &token_ids_end_states {
            map.insert(token_id, end_state);
        }

        {
            return_to.insert(start_state, map);
        }

        // Retrieve the notifier for the current state and notify all waiting threads
        let notifier = {
            Arc::clone(
                state_notifiers
                    .entry(start_state)
                    // technically this should never be done, since if a condvar isnt there then the state should never be computed.
                    // but it shuts-up the compiler.
                    .or_insert_with(|| Arc::new((Mutex::new(false), Condvar::new())))
                    .value(),
            )
        };

        // Set the state to done and notify all waiters
        let (done_lock, condvar) = &*notifier;
        let mut done = done_lock.lock().unwrap();
        *done = true;
        condvar.notify_all();
    });
}

/// Create an FSM state-to-vocabulary map/index through end-to-end token parsing.
///
/// Args:
///     pattern (String): A string pattern to build the DFA.
///     vocabulary (TokenVocabulary): A data structure representing the vocabulary tokens.
///
/// Returns:
///     (BTreeMap<u32, BTreeMap<u32, u32>>, u32, Vec<u32>): A mapping of FSM states to vocabulary token sets,
///     the initial state, and a vector of final states.
/// this feature is in BETA and may not work reliably. try it yourself and see if it works for your regex.
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
