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
#![cfg(feature = "python_bindings")]
use serde::{Serialize, Deserialize};
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
use pyo3::{
    wrap_pyfunction,
    prelude::*,
    exceptions::{
        PyOSError,
        PyValueError,
    },
    types::{
        PyDict,
        PyList
    }
};
use crate::{
    lazy_index::{
        LazyFSMIndex,
        Write,
        Generate
    },
    types::{
        FSMInfo,
    },
    caching::{
        MODULE_STATE,
        get_cached_fsm,
        get_fsm_cache_key,
        CachedFSM,
        start_zmq_thread,
        stop_zmq_thread
    },
    vocab::TokenVocabulary,
};

impl IntoPy<PyObject> for CachedFSM {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new_bound(py);

        let maps_vec: Vec<PyObject> = self.states_to_token_maps
            .into_iter()
            .map(|map| {
                let py_map = PyDict::new_bound(py);
                for (k, v) in map {
                    let _ = py_map.set_item(k, v);
                }
                py_map.into()
            })
            .collect();
        let maps = PyList::new_bound(py, &maps_vec);

        let _ = dict.set_item("states_to_token_maps", maps);
        let _ = dict.set_item("first_state", self.first_state);
        let _ = dict.set_item("finals", &self.finals);
        let _ = dict.set_item("hash", self.hash);

        dict.into()
    }
}

#[pyfunction(name = "get_cached_fsm")]
pub(crate) fn get_cached_fsm_py(py: Python, hash: u64) -> PyResult<PyObject> {
    match get_cached_fsm(hash) {
        Some(cached_fsm_arc) => {
            let cached_fsm = std::sync::Arc::try_unwrap(cached_fsm_arc)
                .unwrap_or_else(|arc| (*arc).clone());
            Ok(cached_fsm.into_py(py)) 
        }
        None => {
            Err(PyValueError::new_err("CachedFSM not found for the given hash"))
        }
    }
}
#[pyfunction(name = "get_fsm_cache_key")]
pub(crate) fn get_fsm_cache_key_py(py: Python, pattern: String, vocab: Py<PyTokenVocabulary>) -> u64 {
    get_fsm_cache_key(&pattern, &vocab.borrow(py).vocab_as_ref())
}

#[derive(Serialize, Deserialize)]
#[pyclass(
    name = "TokenVocabulary",
    module = "faster_outlines.fsm.fsm_utils"
)]
pub struct PyTokenVocabulary {
    pub vocab: TokenVocabulary,
}

impl PyTokenVocabulary {
    pub fn vocab_as_ref(&self) -> &TokenVocabulary {
        &self.vocab
    }
}

#[pymethods]
impl PyTokenVocabulary {
    // We need to allow none arguments, so pickle'ing works.
    /// Initializes the TokenVocabulary from a Python dictionary.
    #[new]
    #[pyo3(signature = (py_dict=None, eos_token_id=None))]
    pub fn new(py_dict: Option<FxHashMap<String, Vec<u32>>>, eos_token_id: Option<u32>) -> PyResult<Self> {
        match (py_dict, eos_token_id) {
            // Normal construction
            (Some(dict), Some(eos)) => {
                let token_vocabulary = TokenVocabulary::from_hashmap(dict, eos);
                Ok(PyTokenVocabulary { vocab: token_vocabulary })
            },
            // Pickle reconstruction (empty instance to be filled by __setstate__)
            (None, None) => {
                Ok(PyTokenVocabulary { 
                    vocab: TokenVocabulary::default()
                })
            },
            _ => {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Both vocab_dict and eos_token_id must be provided, or neither for pickle deserialization"
                ))
            }
        }
    }

    #[getter]
    pub fn eos_token_id(&self) -> PyResult<u32> {
        Ok(self.vocab.eos_token_id)
    }

    /// Adds a token and its values to the vocabulary
    pub fn add_token(&mut self, token: String, values: Vec<u32>) {
        self.vocab.add_token(token, values);
    }

    /// Removes a token from the vocabulary by its name
    pub fn remove_token(&mut self, token: &str) -> Option<Vec<u32>> {
        self.vocab.remove_token(token)
    }

    /// Returns the number of tokens in the vocabulary
    pub fn len(&self) -> usize {
        self.vocab.len()
    }

    /// Checks if the vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let serialized = serde_json::to_string(&self.vocab)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(serialized.as_bytes().into_py(py))
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let bytes: &[u8] = state.extract(py)?;
        let json_str = std::str::from_utf8(bytes)?;
        self.vocab = serde_json::from_str(json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(())
    }
}

#[pyfunction(name = "create_fsm_index_end_to_end_rs")]
#[pyo3(text_signature = "(fsm_info, vocabulary, eos_token_id)")]
pub(crate) fn create_fsm_index_end_to_end_(
    py: Python,
    fsm_info: FSMInfo,
    vocabulary: Py<PyTokenVocabulary>,
) -> LazyFSMIndex {
    let binding = vocabulary.borrow(py);
    let v = binding.vocab_as_ref();
    LazyFSMIndex::new(fsm_info, v, v.eos_token_id)
}

#[pyfunction]
pub(crate) fn start_cache_reciever() -> PyResult<()> {
    start_zmq_thread().map_err(|e| PyOSError::new_err(format!("Failed to start cache receiver: {:?}", e)))
}

#[pyfunction]
pub(crate) fn stop_cache_reciever() -> PyResult<()> {
    stop_zmq_thread().map_err(|e| PyOSError::new_err(format!("Failed to stop cache receiver: {:?}", e)))
}

#[pymodule]
pub fn fsm_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Lazy::force(&MODULE_STATE);
    m.add_function(wrap_pyfunction!(create_fsm_index_end_to_end_, m)?)?;
    m.add_function(wrap_pyfunction!(start_cache_reciever, m)?)?;
    m.add_function(wrap_pyfunction!(stop_cache_reciever, m)?)?;
    m.add_function(wrap_pyfunction!(get_cached_fsm_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_fsm_cache_key_py, m)?)?;
    m.add_class::<LazyFSMIndex>()?;
    m.add_class::<PyTokenVocabulary>()?;
    m.add_class::<Write>()?;
    m.add_class::<Generate>()?;
    Ok(())
}
