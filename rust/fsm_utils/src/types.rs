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

use rustc_hash::FxHashMap;
use std::cell::UnsafeCell;
use std::sync::atomic::AtomicBool;
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

#[derive(Debug, Clone)]
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

#[derive(Clone)]
pub struct Write {
    pub tokens: Vec<i32>,
}

impl Write {
    pub fn new(tokens: Vec<i32>) -> Self {
        Write { tokens }
    }
}

#[derive(Clone)]
pub struct Generate {
    pub tokens: Option<Vec<i32>>,
}

impl Generate {
    pub fn new(tokens: Option<Vec<i32>>) -> Self {
        Generate { tokens }
    }
}

#[derive(Clone)]
pub enum Instruction {
    Write(Write),
    Generate(Generate),
}
