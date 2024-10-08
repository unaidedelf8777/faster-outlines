use pyo3::prelude::*;
use pyo3::{exceptions::PyKeyError, types::PyDict};

use rustc_hash::FxHashMap;

use std::sync::{Arc, Condvar, Mutex};

use std::thread;

use crate::{
    tokenizer_index::create_fsm_index_end_to_end_parallel,
    types::{FSMInfo, TokenVocabulary, ThreadSafeCell},
    caching::{get_cached_fsm, MODULE_STATE, CachedFSM, hash_token_vocabulary}
};

pub(crate) type StateNotifierMap = Arc<Vec<Arc<(Mutex<bool>, Condvar)>>>;

/// Write instruction.
///
/// Attributes
/// ---------
///  - tokens
///     The sequence of tokens to be added to the current sequence by the
///     generation process.
///
#[pyclass]
#[derive(Clone)]
pub struct Write {
    #[pyo3(get, set)]
    pub tokens: Vec<i32>,
}

#[pymethods]
impl Write {
    #[new]
    pub fn new(tokens: Vec<i32>) -> Self {
        Write { tokens }
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Write({:?})", self.tokens))
    }
}

/// Generate instruction
///
/// Attributes
/// ----------
/// - tokens
///     The tokens that lead to a valid completion if generated.  A value
///     of ``None`` indicates that all tokens are allowed.
///
#[pyclass]
#[derive(Clone)]
pub struct Generate {
    #[pyo3(get, set)]
    pub tokens: Option<Vec<i32>>,
}

#[pymethods]
impl Generate {
    #[new]
    pub fn new(tokens: Option<Vec<i32>>) -> Self {
        Generate { tokens }
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Generate({:?})", self.tokens))
    }
}

pub enum Instruction {
    Write { write: Write },
    Generate { generate: Generate },
}

impl IntoPy<PyObject> for Instruction {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            Instruction::Write { write } => {
                let py_write = Py::new(py, write).unwrap();
                py_write.to_object(py)
            }
            Instruction::Generate { generate } => {
                let py_gen = Py::new(py, generate).unwrap();
                py_gen.to_object(py)
            }
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct LazyFSMIndex {
    /// the mapping of states to token subsets from the tokenizer.
    /// this is an interpreted version of the FSM info.
    /// interpreted according to the token vocabulary.
    states_to_token_maps: Arc<Vec<ThreadSafeCell<FxHashMap<u32, u32>>>>,

    /// First state of the FSM
    first_state: u32,

    /// The end-of-sequence token ID from tokenizer.
    eos_token_id: u32,

    // the final states of the fsm
    finals: Vec<u32>,

    // for notifying waiters when a state is finished.
    state_notifiers: StateNotifierMap,

    /// bool indicator, just so we dont need to manually iterate
    /// over the notifiers to check if they are all finished.
    computing_finished: Arc<Mutex<bool>>,
}

/// We gate the creation of new indexes in a impl block seperate to that of the definition
/// of all of the python methods, since the only way to construct a new LazyFSMIndex is through
/// `create_fsm_index_end_to_end_py` function interfaced to python.

impl LazyFSMIndex {
    pub fn new(fsm_info: FSMInfo, vocabulary: TokenVocabulary, eos_token_id: u32) -> Self {

        let cache_key = hash_token_vocabulary(&vocabulary, &fsm_info.pattern);

        let cache_entry = {
            get_cached_fsm(cache_key)
        };

        if let Some(cached_fsm) = cache_entry {
            let states_to_token_maps: Arc<Vec<ThreadSafeCell<FxHashMap<u32,u32>>>> = Arc::new(
                cached_fsm.states_to_token_maps
                    .iter()
                    .map(|map| ThreadSafeCell::new(map.clone()))
                    .collect()
            );

            let state_notifiers = Arc::new(
                (0..states_to_token_maps.len())
                    .map(|_| Arc::new((Mutex::new(true), Condvar::new())))
                    .collect()
            );

            LazyFSMIndex {
                states_to_token_maps,
                first_state: cached_fsm.first_state,
                eos_token_id,
                finals: cached_fsm.finals.clone(),
                computing_finished: Arc::new(Mutex::new(true)),
                state_notifiers,
            }
        } else {
            let results = Arc::new(
                (0..fsm_info.states.len())
                    .map(|_| ThreadSafeCell::new(FxHashMap::default()))
                    .collect::<Vec<_>>(),
            );

            let state_notifiers = Arc::new(
                (0..fsm_info.states.len())
                    .map(|_| Arc::new((Mutex::new(false), Condvar::new())))
                    .collect(),
            );

            let state_notifiers_clone = Arc::clone(&state_notifiers);
            let computing_finished = Arc::new(Mutex::new(false));
            let computing_finished_clone = Arc::clone(&computing_finished);
            let results_clone = Arc::clone(&results);
            let first_state = fsm_info.initial;
            let finals = Arc::new(fsm_info.finals.clone()); 

            let finals_clone = Arc::clone(&finals);

            let cache_key_clone = cache_key;

            thread::spawn(move || {
                create_fsm_index_end_to_end_parallel(
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
                };

                let mut cache = MODULE_STATE.fsm_cache.lock().unwrap();
                cache.put(cache_key_clone, Arc::new(cached_fsm));

                *computing_finished_clone.lock().unwrap() = true;
            });
            let finals = finals.to_vec();
            LazyFSMIndex {
                states_to_token_maps: results,
                first_state,
                eos_token_id,
                finals, 
                computing_finished,
                state_notifiers,
            }
        }
    }

    pub fn get_state_map(&self, state: u32) -> Option<FxHashMap<u32, u32>> {
        // Check if the state index is valid
        if state as usize >= self.states_to_token_maps.len() {
            return None;
        }

        // Retrieve the notifier for the current state
        let notifier = {
            match self.state_notifiers.get(state as usize) {
                Some(notifier_ref) => Arc::clone(notifier_ref),
                None => return None,
            }
        };

        // Wait until the state is computed
        let (done_lock, cvar) = &*notifier;
        let mut done = done_lock.lock().unwrap();
        while !*done {
            done = cvar.wait(done).unwrap();
        }

        // Now it's safe to read from the HashMap
        let cell = &self.states_to_token_maps[state as usize];
        let map = unsafe { &*cell.get_ref() };
        Some(map.clone())
    }
}

// implementation of all the python methods for the LazyFSMIndex struct.
///
/// *LazyFSMIndex*:
///     This struct is a lazy implementation of what the outlines lib normally computes for a regex fsm.
///     It implements both the `Guide` API used by outlines guide objects,
///     and the `HashMap` python api, (.get(key), indexing)
#[pymethods]
impl LazyFSMIndex {
    pub fn get_next_state(&self, state: i32, token_id: u32) -> Option<i32> {
        // check if they are alias states first ( -1, or 0 )
        // state 0 is alias for the first state
        // -1 alias for the last state.
        // if the state is already final, then the next state can only ever be final.
        if state == -1 {
            return Some(-1);
        }

        // Check if the token ID is the end-of-sequence or the state is a final state
        if token_id == self.eos_token_id || self.finals.contains(&(state as u32)) {
            return Some(-1);
        }

        let current_state = if state == 0 {
            self.first_state
        } else {
            state as u32
        };

        // Attempt to find the next state using the get_state_map method
        self.get_state_map(current_state)
            .and_then(|map| map.get(&token_id).copied().map(|s| s as i32))
            .map(|next_state| {
                // If the next state is final, return -1
                if self.is_final_state(next_state) {
                    -1
                } else {
                    next_state
                }
            })
            // If the token to next state pair is not found, return -1 (indicates no valid transition)
            .or(Some(-1))
    }

    pub fn get_next_instruction(&self, state: i32) -> Instruction {
        if self.is_final_state(state) {
            return Instruction::Write {
                write: Write::new(vec![self.eos_token_id as i32]),
            };
        } else {
            match self.get_state_map(state as u32) {
                Some(next_tokens_to_end_states) => {
                    // Collect all keys (token IDs) from the map and convert them to i32
                    let allowed = next_tokens_to_end_states
                        .keys()
                        .cloned()
                        .map(|k| k as i32)
                        .collect::<Vec<i32>>();
                    return Instruction::Generate {
                        generate: Generate::new(Some(allowed)),
                    };
                }
                None => {
                    return Instruction::Write {
                        write: Write::new(vec![self.eos_token_id as i32]),
                    };
                }
            }
        }
    }

    #[inline(always)]
    pub fn is_final_state(&self, state: i32) -> bool {
        // py version:
        // return state == self.final_state

        // Check if the state is the "final" or invalid state
        state == -1 || self.finals.contains(&(state as u32))
    }

    pub fn is_computing_finished(&self) -> bool {
        *self.computing_finished.lock().unwrap()
    }
    /// NOTE: THIS IS NOT VERY PERFORMANT!
    pub fn get_states_to_token_subsets(&self) -> FxHashMap<u32, FxHashMap<u32, u32>> {
        while !self.is_computing_finished() {
            thread::sleep(std::time::Duration::from_millis(2));
        }

        self.states_to_token_maps
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                let map = unsafe { &*cell.get_ref() };
                (index as u32, map.clone())
            })
            .collect()
    }

    pub fn allowed_token_ids(&self, state: i32) -> Vec<i32> {
        if state == -1 {
            return vec![self.eos_token_id as i32];
        }
        match self.get_state_map(state as u32) {
            Some(next_tokens_to_end_states) => {
                // Collect all keys (token IDs) from the map and convert them to i32
                next_tokens_to_end_states
                    .keys()
                    .cloned()
                    .map(|k| k as i32)
                    .collect()
            }
            None => return vec![self.eos_token_id as i32],
        }
    }

    fn get(&self, state: u32, default: Option<PyObject>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            match self.get_state_map(state) {
                Some(map) => {
                    // Convert the HashMap to a Python dict
                    let py_dict = PyDict::new_bound(py);
                    for (k, v) in map.iter() {
                        py_dict.set_item(*k, *v)?;
                    }
                    Ok(py_dict.to_object(py))
                }
                None => match default {
                    Some(default_value) => Ok(default_value),
                    None => Ok(py.None()),
                },
            }
        })
    }

    ///* Python Magic methods *///
    pub fn __repr__(&self) -> PyResult<String> {
        while !self.is_computing_finished() {
            thread::sleep(std::time::Duration::from_millis(100));
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

        Ok(format!(
            "LazyFSMIndex(first_state={}, eos_token_id={}, finals=[{}], states={{{}}})",
            self.first_state, self.eos_token_id, finals, states_display
        ))
    }

    fn __getitem__(&self, state: u32) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            match self.get_state_map(state) {
                Some(map) => {
                    // Convert the HashMap to a Python dict
                    let py_dict = PyDict::new_bound(py);
                    for (k, v) in map.iter() {
                        py_dict.set_item(*k, *v)?;
                    }
                    Ok(py_dict.to_object(py))
                }
                None => Err(PyKeyError::new_err(format!("State {} not found", state))),
            }
        })
    }
}
