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
use serde::{Serialize, Deserialize};
use std::cell::UnsafeCell;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Memory layout for FSM state transition tables.
/// 
/// Structure breakdown:
/// - Arc<Vec<...>>: Shared ownership across threads
/// - ThreadSafeCell: Zero-copy access between threads
/// - FxHashMap<u32,u32>: Individual state transition table
/// 
/// We split the FSM into per-state maps rather than one giant transition table.
/// This approach:
/// 1. Enables parallel computation of different states (we dont do this, but it would be easy to add with say rayon)
/// 2. May improve memory locality (each state's transitions are contiguous) depending on allocator.
/// 3. Avoids large contiguous allocations that could cause fragmentation
pub(crate) type StatesToTokenMaps = Arc<Vec<ThreadSafeCell<FxHashMap<u32, u32>>>>;

/// Thread synchronization primitives for state computation status.
/// 
/// Structure breakdown:
/// - Arc<Vec<...>>: Shared list across threads
/// - Arc<AtomicBool>: Per-state completion flag
/// 
/// Each state gets its own atomic flag that signals when its 
/// transition table is ready. This enables:
/// 1. Fine-grained waiting (threads only block on states they need)
/// 2. Progressive access to completed states
/// 3. Lock-free synchronization via atomic operations
pub(crate) type StateNotifierMap = Arc<Vec<Arc<AtomicBool>>>;

// Zero-copy cross-thread memory access for FSM computation.
// 
// ThreadSafeCell enables the main thread (FSMIndex::new) and computation thread 
// (create_fsm_index_end_to_end) to share memory without copying. This is safe because:
// 
// 1. Only the computation thread writes to the memory
// 2. Main thread only reads after computation is complete
// 3. Atomic flags synchronize access between threads
// 
// While this technically violates Rust's thread safety rules, the manual synchronization
// makes it safe in practice. The performance gain comes from avoiding copies of the
// large state transition maps, which can sometimes have 100's to 1000's of K:V pairs.
#[derive(Debug)] // Debug required because this is wrapped in a Arc<> in usage.
pub(crate) struct ThreadSafeCell<T> {
    value: UnsafeCell<T>,
}

// Tell compiler this type can be shared across threads.
// This is safe due to our manual synchronization.
unsafe impl<T> Sync for ThreadSafeCell<T> {}

impl<T> ThreadSafeCell<T> {
    pub fn new(value: T) -> Self {
        ThreadSafeCell {
            value: UnsafeCell::new(value),
        }
    }

    /// Get mutable access to the inner value.
    /// 
    /// Caller must ensure no other threads are accessing the value.
    /// In our case, this is guaranteed by the atomic flags.
    /// In any other scenario other than ours,
    /// this would be a horrible idea.
    pub unsafe fn get(&self) -> &mut T {
        &mut *self.value.get()
    }

    /// Get read-only access to the inner value.
    /// 
    /// Caller must ensure no threads are writing to the value.
    /// In our case, this is guaranteed by the computing_finished flag.
    pub unsafe fn get_ref(&self) -> &T {
        &*self.value.get()
    }
}

/// FSMInfo implements a Finite State Machine optimized for token sequence matching.
/// Built on interegular's FSM design (https://github.com/MegaIng/interegular),
/// but modified to work with tokenizer outputs instead of raw characters.
///
/// The core idea is converting regex patterns into numeric state machines that can
/// efficiently process token sequences. This differs from traditional regex engines
/// by operating at the token level rather than character level.
///
/// For example, the pattern "[0-9]+" becomes:
/// ```text
/// State 0 (start) --[digit]--> State 1 --[digit]--> State 1 (loop)
///                                      --[EOF]----> State 2 (accept)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FSMInfo {
    /// Starting state for pattern matching
    /// Start state is always 0 in interegular's case,
    /// but it is better not to hardcore that.
    pub initial: u32,

    /// States that represent valid pattern matches. Multiple finals
    /// enable matching patterns like "cat|dog" or "a*", etc.
    pub finals: Vec<u32>,

    /// Fast lookup table for state transitions. Uses a flat map structure
    /// where (current_state, input) -> next_state. For example:
    /// (0, digit_token) -> 1  // Initial digit
    /// (1, digit_token) -> 1  // Additional digits
    /// (1, eof_token)   -> 2  // End of number
    ///
    /// Inside of our algorithms, this usually gets re formatted,
    /// for efficiency and to avoid memory fragmentation. 
    /// (maps can be very large)
    pub transitions: FxHashMap<(u32, u32), u32>,

    /// Maps single UTF-8 characters to their transition keys in the FSM.
    /// Keys are contiguous integers starting from 0 to minimize table size/sparsity.
    /// 
    /// Example for pattern "[ab]c":
    /// "a" -> 0
    /// "b" -> 1  
    /// "c" -> 2
    pub alphabet_symbol_mapping: FxHashMap<String, u32>,

    /// Special transition value for wildcards and catch-alls.
    /// Used by patterns like ".*" or character class negations
    pub alphabet_anything_value: u32,

    /// All valid states in the FSM. Used for pre-allocation and
    /// bounds checking during computation
    ///
    /// We store this because with a large state machine, 
    /// it can take a non-negligible amount of time to count states.
    pub states: Vec<u32>,
    
    /// Source pattern, retained for cache key generation
    pub pattern: String,
}

/// Instructions for controlling LLM token generation.
/// Design inspired by outlines-dev (https://github.com/outlines-dev/outlines)
/// 
/// These instructions map FSM states to allowed token sequences, enabling
/// guided text generation that follows regex patterns.

/// Write instruction for fixed token sequences.
/// Used when an FSM state has a single deterministic path.
/// 
/// Example for pattern "hello":
/// ```text
/// State 0 --[h]--> State 1 --[e]--> State 2 --[l]--> State 3 --[l]--> State 4 --[o]--> Final
/// ```
/// At State 0, we can emit Write([token_id_for_h]) since the path is linear
#[derive(Clone)]
pub struct Write {
    pub tokens: Vec<i32>,
}

impl Write {
    pub fn new(tokens: Vec<i32>) -> Self {
        Write { tokens }
    }
}

/// Generate instruction for branching paths.
/// Used when an FSM state has multiple possible transitions.
/// 
/// Example for pattern "cat|dog":
/// ```text
/// State 0 --[c]--> State 1 --[a]--> State 2 --[t]--> Final
///        \-[d]--> State 3 --[o]--> State 4 --[g]--> Final
/// ```
/// At State 0, we emit Generate([token_id_for_c, token_id_for_d])
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
