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

use std::sync::atomic::AtomicBool;
use std::cell::UnsafeCell;
use rustc_hash::FxHashMap;
#[cfg(feature = "python_bindings")]
use pyo3::prelude::*;
use std::sync::Arc;

pub(crate) type StatesToTokenMaps = Arc<Vec<ThreadSafeCell<FxHashMap<u32, u32>>>>;

pub(crate) type StateNotifierMap = Arc<Vec<Arc<AtomicBool>>>;

// Custom wrapper around UnsafeCell
#[derive(Debug)] // Debug required because this is wrapped in a Arc<> in usage.
pub(crate) struct ThreadSafeCell<T> {
    value: UnsafeCell<T>,
}

// Manually implement Sync
// This seems illegal to do, but its faster this way.
// Not like the compiler will notice :)
unsafe impl<T> Sync for ThreadSafeCell<T> {}

impl<T> ThreadSafeCell<T> {
    pub fn new(value: T) -> Self {
        ThreadSafeCell {
            value: UnsafeCell::new(value),
        }
    }

    pub unsafe fn get(&self) -> &mut T {
        &mut *self.value.get()
    }

    pub unsafe fn get_ref(&self) -> &T {
        &*self.value.get()
    }
}



#[cfg(feature = "python_bindings")]
#[derive(FromPyObject, Debug)]
pub struct FSMInfo {
    /// Initial state of the FSM
    #[pyo3(item("initial"))]
    pub initial: u32,

    /// Final states of the FSM
    #[pyo3(item("finals"))]
    pub finals: Vec<u32>,

    /// The transitions map of the FSM.
    /// Key is a tuple `(state, input)`, where `state` is the current state and `input` is the transition.
    /// Value is the state we will end up at if we take the transition.
    #[pyo3(item("transitions"))]
    pub transitions: FxHashMap<(u32, u32), u32>,

    /// The alphabet mapping.
    /// Key is a `String` representing the input; value is its `u32` identifier / transition key.
    #[pyo3(item("alphabet_symbol_mapping"))]
    pub alphabet_symbol_mapping: FxHashMap<String, u32>,

    #[pyo3(item("alphabet_anything_value"))]
    pub alphabet_anything_value: u32,

    #[pyo3(item("states"))]
    pub states: Vec<u32>,

    /// The pattern used for caching.
    #[pyo3(item("pattern"))]
    pub pattern: String,
}

#[cfg(not(feature = "python_bindings"))]
pub struct FSMInfo {
    /// Initial state of the FSM
    pub initial: u32,

    /// Final states of the FSM
    pub finals: Vec<u32>,

    /// The transitions map of the FSM.
    /// Key is a tuple `(state, input)`, where `state` is the current state and `input` is the transition.
    /// Value is the state we will end up at if we take the transition.
    pub transitions: FxHashMap<(u32, u32), u32>,

    /// The alphabet mapping.
    /// Key is a `String` representing the input; value is its `u32` identifier / transition key.
    pub alphabet_symbol_mapping: FxHashMap<String, u32>,

    pub alphabet_anything_value: u32,

    pub states: Vec<u32>,

    /// The pattern used for caching.
    pub pattern: String,
}



// /// Write instruction.
// ///
// /// Attributes
// /// ---------
// ///  - tokens
// ///     The sequence of tokens to be added to the current sequence by the
// ///     generation process.
// ///
// #[pyclass]
// #[derive(Clone)]
// pub struct Write {
//     #[pyo3(get, set)]
//     pub tokens: Vec<i32>,
// }

// #[pymethods]
// impl Write {
//     #[new]
//     pub fn new(tokens: Vec<i32>) -> Self {
//         Write { tokens }
//     }

//     pub fn __repr__(&self) -> PyResult<String> {
//         Ok(format!("Write({:?})", self.tokens))
//     }
// }

// /// Generate instruction
// ///
// /// Attributes
// /// ----------
// /// - tokens
// ///     The tokens that lead to a valid completion if generated.  A value
// ///     of ``None`` indicates that all tokens are allowed.
// ///
// #[pyclass]
// #[derive(Clone)]
// pub struct Generate {
//     #[pyo3(get, set)]
//     pub tokens: Option<Vec<i32>>,
// }

// #[pymethods]
// impl Generate {
//     #[new]
//     pub fn new(tokens: Option<Vec<i32>>) -> Self {
//         Generate { tokens }
//     }

//     pub fn __repr__(&self) -> PyResult<String> {
//         Ok(format!("Generate({:?})", self.tokens))
//     }
// }

// pub enum Instruction {
//     Write { write: Write },
//     Generate { generate: Generate },
// }

// impl IntoPy<PyObject> for Instruction {
//     fn into_py(self, py: Python) -> PyObject {
//         match self {
//             Instruction::Write { write } => {
//                 let py_write = Py::new(py, write).unwrap();
//                 py_write.to_object(py)
//             }
//             Instruction::Generate { generate } => {
//                 let py_gen = Py::new(py, generate).unwrap();
//                 py_gen.to_object(py)
//             }
//         }
//     }
// }
