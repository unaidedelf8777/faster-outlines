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
use crate::{
    atomic_wait::platform::wake_all,
    types::{FSMInfo, StateNotifierMap, StatesToTokenMaps},
    vocab::TokenVocabulary,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[inline(always)]
fn create_vocab_transition_vector(
    alphabet_symbol_mapping: &FxHashMap<char, u32>,
    alphabet_anything_value: u32,
    vocabulary: &Vec<(String, Vec<u32>)>,
) -> Vec<Vec<u32>> {
    vocabulary
        .iter()
        .map(|(token_str, _)| {
            token_str
                .chars()
                .map(|c| {
                    *alphabet_symbol_mapping
                        .get(&c)
                        .unwrap_or(&alphabet_anything_value)
                })
                .collect()
        })
        .collect()
}

fn walk_fsm(
    fsm_info: &FSMInfo,
    token_transition_keys: &[u32],
    start_state: u32,
    full_match: bool,
) -> Vec<u32> {
    let mut state = start_state;
    let mut accepted_states = Vec::new();
    let mut last_final_idx = 0;

    for (i, &trans_key) in token_transition_keys.iter().enumerate() {
        match fsm_info.transitions.get(&(state, trans_key)) {
            Some(&new_state) => {
                state = new_state;
                if fsm_info.finals.contains(&state) {
                    last_final_idx = i + 1;
                }
                accepted_states.push(state);
            }
            None => {
                if !full_match && last_final_idx > 0 {
                    return accepted_states[..last_final_idx].to_vec();
                }
                return Vec::new();
            }
        }
    }

    if full_match && last_final_idx != token_transition_keys.len() {
        return Vec::new();
    }

    accepted_states
}

/// Maps a single FSM state to its valid token transitions.
/// 
/// For each vocabulary token:
/// 1. Walks the FSM using the token's transition sequence
/// 2. If walk succeeds (partial or full), records (token_id, end_state) pair
/// 
/// # Arguments
/// * `fsm_info` - FSM definition with transitions and final states
/// * `vocabulary` - List of token IDs for each vocab entry
/// * `vocabulary_transition_keys` - Pre-computed transition sequences for each token
/// * `start_state` - State to compute transitions for
/// 
/// # Returns
/// Set of (token_id, end_state) pairs representing valid transitions
/// 
/// # Example
/// For pattern "[a-c]+" at state 0:
/// ```text
/// Vocabulary:
/// - "a" -> [7]     transition_keys: [0]
/// - "b" -> [8]     transition_keys: [1]  
/// - "c" -> [9]     transition_keys: [2]
/// - "ab" -> [15]   transition_keys: [0,1]
/// - "bc" -> [16]   transition_keys: [1,2]
/// 
/// FSM walks:
/// "a":  0 --[0]--> 1  ✓ Add (7,1)
/// "b":  0 --[1]--> 1  ✓ Add (8,1)
/// "c":  0 --[2]--> 1  ✓ Add (9,1)
/// "ab": 0 --[0]--> 1 --[1]--> 1  ✓ Add (15,1)
/// "bc": 0 --[1]--> 1 --[2]--> 1  ✓ Add (16,1)
/// 
/// Returns: {(7,1), (8,1), (9,1), (15,1), (16,1)}
/// ```
/// Note this walking example is naive, and does not account for anything_else_value,
/// or a few other conditions, such as final state logic, but it gets the point across.
fn state_scan_tokens(
    fsm_info: &FSMInfo,
    vocabulary: &[Vec<u32>],
    vocabulary_transition_keys: &[Vec<u32>],
    start_state: u32,
) -> FxHashSet<(u32, u32)> {
    vocabulary
        .iter()
        .zip(vocabulary_transition_keys.iter())
        .flat_map(|(token_ids, token_transition_keys)| {
            let state_seq = walk_fsm(fsm_info, token_transition_keys, start_state, false);
            let last_state_opt = if state_seq.len() < token_transition_keys.len() {
                None
            } else {
                Some(*state_seq.last().unwrap())
            };
            token_ids.iter().filter_map(move |&token_id| {
                last_state_opt.map(|last_state| (token_id, last_state))
            })
        })
        .collect::<FxHashSet<(u32, u32)>>()
}

/// Core FSM computation function that builds token transition maps.
/// 
/// # Memory Layout
/// The function receives shared memory structures from LazyFSMIndex:
/// - return_to: Pre-allocated state transition tables (Arc<Vec<ThreadSafeCell>>)
/// - state_notifiers: Atomic flags for completion status (Arc<Vec<Arc<AtomicBool>>>)
/// 
/// # Processing Flow
/// 1. Setup Phase:
///    - Converts vocabulary into numeric transition keys
///    - Strips string data to minimize memory during computation
///    - Each token gets mapped to its FSM transition sequence
/// 
/// 2. State Processing:
///    - Processes each FSM state independently
///    - For each state:
///      a. Simulates FSM walks for all vocabulary tokens
///      b. Records valid (token_id, end_state) pairs
///      c. Writes results directly to shared memory
///      d. Signals completion via atomic flag
/// 
/// # Memory Safety
/// - Writes to shared memory are safe because:
///   1. Each state's map is accessed only by one thread
///   2. ThreadSafeCell provides zero-copy access
///   3. Atomic flags synchronize readers/writer
///
/// # Example Flow
/// For pattern "[a-c]+" and vocabulary:
/// ```text
/// {
///   "a": [7],    // token ID for 'a'
///   "b": [8],    // token ID for 'b'
///   "c": [9],    // token ID for 'c'
///   "ab": [15],  // token ID for string "ab"
///   "bc": [16]   // token ID for string "bc"
/// }
/// ```
/// 
/// FSM structure:
/// ```text
/// State 0 (start) --[a,b,c]--> State 1 (accept) --[a,b,c]--> State 1 (loop)
/// 
/// transitions = {
///   (0,'a') -> 1,  // 'a' is transition key 0
///   (0,'b') -> 1,  // 'b' is transition key 1
///   (0,'c') -> 1,  // 'c' is transition key 2
///   (1,'a') -> 1,
///   (1,'b') -> 1,
///   (1,'c') -> 1
/// }
/// ```
/// 
/// Processing flow:
/// ```text
/// 1. Setup:
///    alphabet_symbol_mapping = {"a": 0, "b": 1, "c": 2}
///    Create transition keys for each vocab token:
///      "a"  -> [0]     // single char 'a' maps to transition 0
///      "b"  -> [1]     // single char 'b' maps to transition 1
///      "c"  -> [2]     // single char 'c' maps to transition 2
///      "ab" -> [0, 1]  // chars 'a','b' map to transitions 0,1
///      "bc" -> [1, 2]  // chars 'b','c' map to transitions 1,2
/// 
/// 2. Process State 0:
///    Walk FSM for each token's transition sequence:
///    - "a" [0]     -> ends in state 1 ✓
///    - "b" [1]     -> ends in state 1 ✓
///    - "c" [2]     -> ends in state 1 ✓
///    - "ab" [0,1]  -> no valid path (multi-char not accepted here)
///    - "bc" [1,2]  -> no valid path (multi-char not accepted here)
///    Write to state 0's map: {7->1, 8->1, 9->1}
///    Signal state 0 complete via futex
/// 
/// 3. Process State 1:
///    Walk FSM for each token:
///    - "a" [0]     -> ends in state 1 ✓
///    - "b" [1]     -> ends in state 1 ✓
///    - "c" [2]     -> ends in state 1 ✓
///    - "ab" [0,1]  -> ends in state 1 ✓
///    - "bc" [1,2]  -> ends in state 1 ✓
///    Write to state 1's map: {7->1, 8->1, 9->1}
///    Signal state 1 complete via futex
///     
///    Note how even though "ab", and "bc" are multiple transitions, they are
///    Still accepted because they still follow transitions which are valid
///    for state 1 ( [a-c]+ ).
/// ```
pub(crate) fn create_fsm_index_end_to_end(
    fsm_info: &FSMInfo,
    vocabulary: &TokenVocabulary,
    return_to: &StatesToTokenMaps,
    state_notifiers: &StateNotifierMap,
) {
    let vocabulary = vocabulary
        .into_iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect::<Vec<(String, Vec<u32>)>>();
    
    let alphabet_symbol_mapping: FxHashMap<char, u32> = fsm_info
        .alphabet_symbol_mapping
        .iter()
        .map(|(k, &v)| (k.chars().next().unwrap(), v))
        .collect();

    let vocabulary_transition_keys = create_vocab_transition_vector(
        &alphabet_symbol_mapping,
        fsm_info.alphabet_anything_value,
        &vocabulary,
    );

    let vocabulary_entries_only_values: Vec<Vec<u32>> = vocabulary
        .into_iter()
        .map(|(_, v)| v.clone()) // Remove the String and retain Vec<u32>, to reduce mem usage.
        .collect();

    fsm_info.states.iter().for_each(|&start_state| {
        let token_ids_end_states = state_scan_tokens(
            fsm_info,
            &vocabulary_entries_only_values,
            &vocabulary_transition_keys,
            start_state,
        );

        unsafe {
            let map = return_to[start_state as usize].get();
            for (token_id, end_state) in token_ids_end_states {
                map.insert(token_id, end_state);
            }
        }

        let notifier = Arc::clone(&state_notifiers[start_state as usize]);
        let atomic = notifier;
        atomic.store(true, Ordering::Release);
        wake_all(&*atomic)
    });
}
