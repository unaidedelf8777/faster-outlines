/// * Imports * ///
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

//use crate::caching::get_or_create_vocab_trie;
use crate::lazy_index::StateNotifierMap;
use crate::types::{FSMInfo, ThreadSafeCell, TokenVocabulary};
use crate::lazy_index::LazyFSMIndex;

fn get_vocabulary_transition_keys(
    alphabet_symbol_mapping: &FxHashMap<String, u32>,
    alphabet_anything_value: u32,
    vocabulary: &[(String, Vec<u32>)],
) -> Vec<Vec<u32>> {
    let mut vocab_transition_keys: Vec<Vec<u32>> = Vec::new();

    for (token_str, _) in vocabulary.iter() {
        let token_transition_keys = 
            get_token_transition_keys(
                alphabet_symbol_mapping,
                alphabet_anything_value,
                token_str,
            );

        vocab_transition_keys.push(token_transition_keys);
    }

    vocab_transition_keys
}

fn get_token_transition_keys(
    alphabet_symbol_mapping: &FxHashMap<String, u32>,
    alphabet_anything_value: u32,
    token_str: &str,
) -> Vec<u32> {
    let mut token_transition_keys = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = token_str.chars().collect();

    while i < chars.len() {
        let symbol;
        if chars[i] == '\0' && i + 2 < chars.len() {
            symbol = format!("\0{}{}", chars[i + 1], chars[i + 2]);
            i += 3;
        } else {
            symbol = chars[i].to_string();
            i += 1;
        }

        let transition_key = *alphabet_symbol_mapping
            .get(&symbol)
            .unwrap_or(&alphabet_anything_value);
        token_transition_keys.push(transition_key);
    }

    token_transition_keys
}

fn walk_fsm(
    fsm_transitions: &FxHashMap<(u32, u32), u32>,
    _fsm_initial: u32,
    fsm_finals: &Arc<Vec<u32>>,
    token_transition_keys: &[u32],
    start_state: u32,
    full_match: bool,
) -> Vec<u32> {
    let mut state = start_state;
    let mut accepted_states = Vec::new();
    let mut last_final_idx = 0;

    for (i, &trans_key) in token_transition_keys.iter().enumerate() {
        match fsm_transitions.get(&(state, trans_key)) {
            Some(&new_state) => {
                state = new_state;
                if fsm_finals.contains(&state) {
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

fn state_scan_tokens(
    fsm_transitions: &FxHashMap<(u32, u32), u32>,
    fsm_initial: u32,
    fsm_finals: &Arc<Vec<u32>>,
    vocabulary: &[(String, Vec<u32>)],
    vocabulary_transition_keys: &[Vec<u32>],
    start_state: u32,
) -> FxHashSet<(u32, u32)> {
    let mut res = FxHashSet::default();

    for ((_, token_ids), token_transition_keys) in
        vocabulary.iter().zip(vocabulary_transition_keys.iter())
    {
        let state_seq = walk_fsm(
            fsm_transitions,
            fsm_initial,
            fsm_finals,
            token_transition_keys,
            start_state,
            false,
        );

        if state_seq.len() < token_transition_keys.len() {
            continue;
        }

        let last_state = *state_seq.last().unwrap();
        for &token_id in token_ids {
            res.insert((token_id, last_state));
        }
    }

    res
}

pub fn create_fsm_index_end_to_end_parallel(
    fsm_info: &FSMInfo,
    vocabulary: &TokenVocabulary,
    return_to: &Arc<Vec<ThreadSafeCell<FxHashMap<u32, u32>>>>,
    state_notifiers: &StateNotifierMap, 
) {

    let vocabulary_entries: Vec<(String, Vec<u32>)> = vocabulary
        .iter()
        .map(|(s, v)| (s.clone(), v.clone()))
        .collect();

    let vocabulary_transition_keys = get_vocabulary_transition_keys(
        &fsm_info.alphabet_symbol_mapping,
        fsm_info.alphabet_anything_value,
        &vocabulary_entries,
    );
    
    let vocabulary_transition_keys = Arc::new(vocabulary_transition_keys);
    let fsm_transitions = Arc::new(fsm_info.transitions.clone());
    let fsm_finals = Arc::new(fsm_info.finals.clone()); 

    fsm_info.states.par_iter().for_each(|&start_state| {
        let token_ids_end_states = state_scan_tokens(
            &fsm_transitions,
            fsm_info.initial,
            &fsm_finals,
            &vocabulary_entries,
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
    fsm_info: FSMInfo,
    vocabulary: TokenVocabulary,
    eos_token_id: u32,
) -> LazyFSMIndex {
    LazyFSMIndex::new(fsm_info, vocabulary, eos_token_id)
}
