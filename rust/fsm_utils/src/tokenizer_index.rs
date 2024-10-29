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
