/* The MIT License (MIT)
* Copyright (c) 2024 Nathan Hoos
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

#![cfg(feature = "python_bindings")]
// Serde is implemented on data classes for compatibility with
// multi-python interpreter inference engines like VLLM
use serde::{Serialize, Deserialize};
use once_cell::sync::Lazy;
use rustc_hash::{FxHashMap, FxHashSet};
use anyhow::{Result, Context};

use pyo3::{
    wrap_pyfunction,
    prelude::*,
    exceptions::{
        PyRuntimeError,
        PyValueError,
    }
};
use crate::{
    lazy_index::{
        LazyFSMIndex
    },
    caching::{
        MODULE_STATE
    },
    types::{
        Write,
        Generate,
        Instruction,
        FSMInfo,
        TransitionMap
    },
    vocab::TokenVocabulary,
};

#[derive(Serialize, Deserialize)]
#[pyclass(
    name = "TokenVocabulary",
    module = "faster_outlines.lib"
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
    #[pyo3(signature = (py_dict=None, eos_token_id=None, special_tokens=None))]
    pub fn new(py_dict: Option<FxHashMap<String, u32>>, eos_token_id: Option<u32>, special_tokens: Option<FxHashSet<String>>) -> PyResult<Self> {
        match (py_dict, eos_token_id, special_tokens) {
            // Normal construction
            (Some(dict), Some(eos), Some(special)) => {
                let token_vocabulary = TokenVocabulary::from_raw_vocab(dict, eos, Some(special), Some(true))
                    .with_context(|| format!("Failed to create token vocabulary due to error."))?;
                Ok(PyTokenVocabulary { vocab: token_vocabulary })
            },
            // Pickle reconstruction (empty instance to be filled by __setstate__)
            (None, None, None) => {
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

#[pyclass(name = "Write")]
pub struct PyWrite {
    #[pyo3(get, set)]
    pub tokens: Vec<i32>,
}

#[pymethods]
impl PyWrite {
    #[new]
    pub fn new(tokens: Vec<i32>) -> Self {
        PyWrite { tokens }
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Write({:?})", self.tokens))
    }
}

impl From<Write> for PyWrite {
    fn from(write: Write) -> Self {
        PyWrite { tokens: write.tokens }
    }
}

#[pyclass(name = "Generate")]
pub struct PyGenerate {
    #[pyo3(get, set)]
    pub tokens: Option<Vec<i32>>,
}

#[pymethods]
impl PyGenerate {
    #[new]
    #[pyo3(signature = (tokens=None))]
    pub fn new(tokens: Option<Vec<i32>>) -> Self {
        PyGenerate { tokens }
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Generate({:?})", self.tokens))
    }
}

impl From<Generate> for PyGenerate {
    fn from(generate: Generate) -> Self {
        PyGenerate { tokens: generate.tokens }
    }
}

impl IntoPy<PyObject> for Instruction {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            Instruction::Write(write) => {
                let py_write: PyWrite = write.into();
                py_write.into_py(py)
            }
            Instruction::Generate(generate) => {
                let py_generate: PyGenerate = generate.into();
                py_generate.into_py(py)
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[pyclass(
    name = "FSMInfo"
)]
pub struct PyFSMInfo(FSMInfo);

#[pymethods]
impl PyFSMInfo {
    #[new]
    pub fn new(
        initial: u32,
        finals: Vec<u32>,
        transitions: FxHashMap<(u32,u32), u32>,
        alphabet_symbol_mapping: FxHashMap<String, u32>,
        alphabet_anything_value: u32,
        pattern: String
    ) -> Self {
        let transitions_map: TransitionMap = transitions.into();
        PyFSMInfo(FSMInfo {
            initial: initial,
            finals: finals,
            transitions: transitions_map,
            alphabet_symbol_mapping: alphabet_symbol_mapping,
            alphabet_anything_value: alphabet_anything_value,
            pattern: pattern,
        })
    }

    #[getter]
    pub fn initial(&self) -> u32 {
        self.0.initial
    }

    #[getter]
    pub fn finals(&self) -> Vec<u32> {
        self.0.finals.clone() 
    }

    #[getter]
    pub fn alphabet_symbol_mapping(&self) -> FxHashMap<String, u32> {
        self.0.alphabet_symbol_mapping.clone()
    }

    #[getter]
    pub fn alphabet_anything_value(&self) -> u32 {
        self.0.alphabet_anything_value
    }

    #[getter]
    pub fn pattern(&self) -> String {
        self.0.pattern.clone()
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let serialized = serde_json::to_string(&self.0)
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;
        Ok(serialized.as_bytes().into_py(py))
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        let bytes: &[u8] = state.extract(py)?;
        let json_str = std::str::from_utf8(bytes)?;
        self.0 = serde_json::from_str(json_str)
            .map_err(|e| PyErr::new::<PyValueError, _>(e.to_string()))?;
        Ok(())
    }
}

impl From<PyFSMInfo> for FSMInfo {
    fn from(pyfsm: PyFSMInfo) -> FSMInfo {
        pyfsm.0
    }
}



#[pyclass(name = "LazyFSMIndex")]
#[derive(Clone)]
pub struct PyLazyFSMIndex {
    inner: LazyFSMIndex
}

impl PyLazyFSMIndex {
    pub fn new(
        fsm_info: FSMInfo,
        vocabulary: &TokenVocabulary
    ) -> Result<Self> {
        Ok(PyLazyFSMIndex {
            inner: LazyFSMIndex::new(
                fsm_info,
                vocabulary, 
                vocabulary.eos_token_id
            )
        })
    }
}

#[pymethods]
impl PyLazyFSMIndex {
    

    pub fn get_next_state(&self, state: i32, token_id: u32) -> Option<i32> {
        self.inner.get_next_state(state, token_id)
    }

    pub fn get_next_instruction(&self, state: i32) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let instruction = self.inner.get_next_instruction(state);
            match instruction {
                Instruction::Write(write) => {
                    let py_write: PyWrite = write.into();
                    Ok(py_write.into_py(py))
                },
                Instruction::Generate(generate) => {
                    let py_generate: PyGenerate = generate.into();
                    Ok(py_generate.into_py(py))
                }
            }
        })
    }

    pub fn collect_finished_states(&mut self) -> PyResult<FxHashMap<u32, FxHashMap<u32, u32>>> {
        self.inner.collect_finished_states()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn await_state(&self, state_index: u32) -> PyResult<()> {
        self.inner.await_state(state_index)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn await_finished(&self) {
        self.inner.await_finished()
    }

    pub fn get_allowed_token_ids(&self, state: i32) -> Vec<i32> {
        self.inner.get_allowed_token_ids(state)
    }

    pub fn __repr__(&self) -> String {
        self.inner.__repr__()
    }
}

#[pyfunction(name = "create_fsm_index_end_to_end_rs")]
#[pyo3(text_signature = "(fsm_info, vocabulary, eos_token_id)")]
pub(crate) fn create_fsm_index_end_to_end_<'py>(
    py: Python<'py>,
    fsm_info: PyFSMInfo,
    vocabulary: Py<PyTokenVocabulary>,
) -> PyResult<PyLazyFSMIndex> {
    let f: FSMInfo = fsm_info.into();
    let v = vocabulary.borrow(py);
    let v = v.vocab_as_ref();
    let result: Result<PyLazyFSMIndex> = (|| {
        
        PyLazyFSMIndex::new(f, v)
            .context("Failed to create FSM index")
    })();

    result.map_err(|e| {
        PyRuntimeError::new_err(format!("FSM index creation failed: {:#}", e))
    })
}

#[pymodule]
pub fn lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Lazy::force(&MODULE_STATE);
    m.add_function(wrap_pyfunction!(create_fsm_index_end_to_end_, m)?)?;

    m.add_class::<PyFSMInfo>()?;
    m.add_class::<PyLazyFSMIndex>()?;
    m.add_class::<PyTokenVocabulary>()?;
    m.add_class::<PyWrite>()?;
    m.add_class::<PyGenerate>()?;
    Ok(())
}
