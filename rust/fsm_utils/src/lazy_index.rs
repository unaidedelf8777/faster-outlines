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
use crate::types::{StateNotifierMap, StatesToTokenMaps};
use crate::{
    atomic_wait::platform::{wait, wake_all},
    caching::{get_cached_fsm, get_fsm_cache_key, insert_fsm_to_cache, CachedFSM},
    tokenizer_index::create_fsm_index_end_to_end,
    types::{FSMInfo, Generate, Instruction, ThreadSafeCell, Write},
    vocab::TokenVocabulary,
};
use anyhow::Result;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use fixedbitset::FixedBitSet;

/// LazyFSMIndex implements a lazy-loading finite state machine (FSM) for efficient token sequence matching.
/// It processes state transitions asynchronously and caches results for improved performance.
///
/// # Memory Architecture
/// - Uses thread-safe shared memory structures
/// - State transitions stored in pre-allocated tables
/// - Atomic flags coordinate computation status
///
/// # Concurrency Model
/// ```text
/// Main Thread                    Compute Thread
/// ───────────                    ──────────────
/// new() ──────┐                  ┌── Start computation
///             │                  │   For each state:
/// get_state() │◄──── Atomic ────►│   1. Process tokens
///             │      Flags       │   2. Write results
///             │                  │   3. Signal done
/// await() ────┘                  └── Cache results
/// ```
#[derive(Clone)]
pub struct LazyFSMIndex {
    /// The mapping of states to token subsets from the tokenizer.
    states_to_token_maps: StatesToTokenMaps,

    /// Entry point state ID.
    /// Must be valid state in FSM info.
    first_state: u32,

    /// The end-of-sequence token ID from tokenizer.
    eos_token_id: u32,

    /// the final states of the fsm
    finals: Vec<u32>,

    /// For notifying waiters when a state is finished.
    state_notifiers: StateNotifierMap,

    /// bool indicator, just so we dont need to manually iterate
    /// over the notifiers to check if they are all finished.
    computing_finished: Arc<AtomicBool>,

    returned_states: FixedBitSet
}

// This impl block holds all methods which are not feature specific,
// Other impl blocks are specific to where the object is being used from ( i.e. python, rust )
impl LazyFSMIndex {
    pub fn new<'v>(fsm_info: FSMInfo, vocabulary: &'v TokenVocabulary, eos_token_id: u32) -> Self {
        let vocabulary = vocabulary.clone();
        let cache_key = get_fsm_cache_key(&fsm_info.pattern, &vocabulary);

        let cache_entry = { get_cached_fsm(cache_key) };

        match cache_entry {
            Some(cached_fsm) => {
                let states_to_token_maps: Arc<Vec<ThreadSafeCell<FxHashMap<u32, u32>>>> = Arc::new(
                    cached_fsm
                        .states_to_token_maps
                        .iter()
                        .map(|map| ThreadSafeCell::new(map.clone()))
                        .collect(),
                );

                let state_notifiers: StateNotifierMap = Arc::new(
                    (0..fsm_info.states.len())
                        .map(|_| Arc::new(AtomicBool::new(true)))
                        .collect(),
                );
                let returned_states_set = FixedBitSet::with_capacity(fsm_info.states.len());

                let fsm_index = LazyFSMIndex {
                    states_to_token_maps: states_to_token_maps,
                    first_state: cached_fsm.first_state,
                    eos_token_id: eos_token_id,
                    finals: cached_fsm.finals.clone(),
                    computing_finished: Arc::new(AtomicBool::new(true)),
                    state_notifiers: state_notifiers,
                    returned_states: returned_states_set,
                };

                return fsm_index;
            }
            None => {
                let results: Arc<Vec<ThreadSafeCell<FxHashMap<u32, u32>>>> = Arc::new(
                    (0..fsm_info.states.len())
                        .map(|_| ThreadSafeCell::new(FxHashMap::default()))
                        .collect::<Vec<_>>(),
                );

                let state_notifiers: StateNotifierMap = Arc::new(
                    (0..fsm_info.states.len())
                        .map(|_| Arc::new(AtomicBool::new(false)))
                        .collect(),
                );

                let state_notifiers_clone = Arc::clone(&state_notifiers);
                let computing_finished = Arc::new(AtomicBool::new(false));
                let computing_finished_clone = Arc::clone(&computing_finished);
                let results_clone = Arc::clone(&results);
                let first_state = fsm_info.initial;
                let finals = Arc::new(fsm_info.finals.clone());
                let finals_clone = Arc::clone(&finals);
                let cache_key_clone = cache_key;
                let returned_states_set = FixedBitSet::with_capacity(fsm_info.states.len());

                thread::spawn(move || {
                    create_fsm_index_end_to_end(
                        &fsm_info,
                        &vocabulary,
                        &results_clone,
                        &state_notifiers_clone,
                    );
                    let cached_fsm = CachedFSM {
                        states_to_token_maps: results_clone
                            .iter()
                            .map(|cell| unsafe { &*cell.get_ref() }.clone())
                            .collect(),
                        first_state,
                        finals: finals_clone.to_vec(),
                        hash: cache_key_clone.clone(),
                    };
                    insert_fsm_to_cache(cached_fsm, cache_key_clone);
                    computing_finished_clone.store(true, Ordering::Release);
                    wake_all(&*computing_finished_clone);
                });
                let finals = finals.to_vec();
                LazyFSMIndex {
                    states_to_token_maps: results,
                    first_state: first_state,
                    eos_token_id: eos_token_id,
                    finals: finals,
                    computing_finished: computing_finished,
                    state_notifiers: state_notifiers,
                    returned_states: returned_states_set,
                }
            }
        }
    }

    /// Retrieves token transition map for state.
    /// Blocks if state computation pending.
    ///
    /// # Memory Safety
    /// - Reader synchronized via atomic flag
    /// - Zero-copy access through ThreadSafeCell
    /// - Single writer guarantee from compute thread
    /// - Immutable after computation finishes
    ///
    /// # Performance
    /// - O(1) access after computation
    /// - Blocking if state pending
    fn get_state_map(&self, state: u32) -> Option<&FxHashMap<u32, u32>> {
        if state as usize >= self.states_to_token_maps.len() {
            return None;
        }

        let notifier = {
            match self.state_notifiers.get(state as usize) {
                Some(notifier_ref) => notifier_ref,
                None => return None,
            }
        };

        let atomic = &**notifier;
        wait(&atomic, false); // if the value is false, wait.

        let cell = &self.states_to_token_maps[state as usize];
        Some(unsafe { &*cell.get_ref() })
    }

    /// Tests if state represents pattern match.
    ///
    /// # Special States
    /// - -1: Universal terminal state
    /// - 0: Initial state alias
    /// - finals: Pattern-specific terminals
    ///
    #[inline(always)]
    fn is_final_state(&self, state: i32) -> bool {
        // Check if the state is the "final" or invalid state
        state == -1 || self.finals.contains(&(state as u32))
    }

    /// Checks global computation status.
    #[inline(always)]
    fn is_computing_finished(&self) -> bool {
        self.computing_finished.load(Ordering::Acquire)
    }
}

// All public methods.
impl LazyFSMIndex {
    /// Based on the current state + token_id ( transition ) generated,
    /// figure out what state in the FSM this state + transition would arrive at.
    pub fn get_next_state(&self, state: i32, token_id: u32) -> Option<i32> {
        // Handle special states (-1 is final state, 0 is the first state alias)
        if state == -1 {
            return Some(-1);
        }

        if token_id == self.eos_token_id || self.finals.contains(&(state as u32)) {
            return Some(-1);
        }

        let current_state = if state == 0 {
            self.first_state
        } else {
            state as u32
        };

        if let Some(map) = self.get_state_map(current_state) {
            if let Some(&next_state_u32) = map.get(&token_id) {
                let next_state = next_state_u32 as i32;
                if self.is_final_state(next_state) {
                    Some(-1)
                } else {
                    Some(next_state)
                }
            } else {
                Some(-1)
            }
        } else {
            Some(-1)
        }
    }

    /// Generates next pattern-matching instruction.
    ///
    /// # Instructions
    /// - Write([tokens]): Fixed sequence
    /// - Generate(Some([tokens])): Constrained choice
    /// - Generate(None): Unconstrained
    ///
    /// # State Handling
    /// - Terminal: Write([EOS])
    /// - Valid: Generate(allowed_tokens)
    /// - Invalid: Write([EOS])
    ///
    pub fn get_next_instruction(&self, state: i32) -> Instruction {
        if self.is_final_state(state) {
            return Instruction::Write(Write::new(vec![self.eos_token_id as i32]));
        }

        let current_state = if state == 0 {
            self.first_state
        } else {
            state as u32
        };

        if let Some(map) = self.get_state_map(current_state) {
            let allowed = map.keys().cloned().map(|k| k as i32).collect::<Vec<i32>>();
            Instruction::Generate(Generate::new(Some(allowed)))
        } else {
            Instruction::Write(Write::new(vec![self.eos_token_id as i32]))
        }
    }

    /// Blocks until specific state completes
    /// computation, and can be retrieved.
    ///
    /// # Errors
    /// - State index out of bounds
    /// - State not scheduled for computation
    pub fn await_state(&self, state_index: u32) -> Result<()> {
        if (state_index as usize) >= self.states_to_token_maps.len() {
            bail!(
                "State {} is not in computed states, and is not set to be computed. Does this state exist?",
                state_index
            );
        }

        let notifier = &self.state_notifiers[state_index as usize];
        let atomic = &**notifier;
        wait(&atomic, false);
        Ok(())
    }

    /// Blocks until all states finish.
    pub fn await_finished(&self) {
        wait(&self.is_computing_finished, false);
    }

    /// Collects newly computed state transitions.
    /// 
    /// This is an api which takes no arguments, and is useful for people building on top of 
    /// the computed transitions computation of `LazyFSMIndex` who want access to the raw state transitions
    /// map in realtime, while it is being computed.
    pub fn collect_finished_states(&mut self) -> Result<FxHashMap<u32, FxHashMap<u32, u32>>> {
        let mut finished_states = FxHashMap::default();
        let total_states = self.states_to_token_maps.len();
        
        for index in 0..total_states {
            if self.returned_states.contains(index) {
                continue;
            }

            let state_is_done = {
                let notifier = &self.state_notifiers[index];
                let atomic = &**notifier;
                atomic.load(Ordering::Acquire)
            };

            if state_is_done {
                if let Some(state_map) = self.get_state_map(index as u32) {
                    finished_states.insert(index as u32, state_map.clone());
                    self.returned_states.set(index, true);
                } else {
                }
            } else {
            }
        }
        Ok(finished_states)
    }

    /// Retrieve a vector of allowed Token ID's at the state `state`
    ///
    /// This is an alternative to the Instruction based API used
    /// in `get_next_instruction`. The Instruction based API is
    /// preffered, but this can be useful for debugging or more manual
    /// implementations / logic about state transition / token ID selection.
    pub fn get_allowed_token_ids(&self, state: i32) -> Vec<i32> {
        if state == -1 {
            return vec![self.eos_token_id as i32];
        }
        match self.get_state_map(state as u32) {
            Some(next_tokens_to_end_states) => next_tokens_to_end_states
                .keys()
                .cloned()
                .map(|k| k as i32)
                .collect(),
            None => return vec![self.eos_token_id as i32],
        }
    }

    ///* Python Magic methods *///
    /// WARNING: THIS WILL BLOCK UNTIL FSM IS FINISHED COMPUTING!
    pub fn __repr__(&self) -> String {
        while !self.is_computing_finished() {
            thread::sleep(std::time::Duration::from_millis(1));
        }

        let states: String = self
            .states_to_token_maps
            .iter()
            .take(10)
            .enumerate()
            .map(|(index, cell)| {
                let state_map = unsafe { &*cell.get_ref() };
                format!("{}: {:?}", index, state_map)
            })
            .collect::<Vec<String>>()
            .join(", ");

        let states_display = if self.states_to_token_maps.len() > 10 {
            format!("{}, ...", states)
        } else {
            states
        };

        let finals = self
            .finals
            .iter()
            .map(|state| state.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        format!(
            "LazyFSMIndex(first_state={}, eos_token_id={}, finals=[{}], states={{{}}})",
            self.first_state, self.eos_token_id, finals, states_display
        )
    }
}
