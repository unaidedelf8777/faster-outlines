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
use once_cell::sync::Lazy;
use std::env;

/// Maximum number of FSM patterns to cache.
///
/// # Environment Configuration
/// Set via `FASTER_OUTLINES_CACHE_SIZE` environment variable.
///
/// # Default Behavior
/// - Default size: 100 patterns
/// - Implements LRU eviction policy
/// - Thread-safe access via Lazy initialization
///
/// Set the env var like so:
/// ```bash
/// export FASTER_OUTLINES_CACHE_SIZE=[INTEGER]
/// ```
///
/// # Memory Impact
/// Cache size directly affects memory usage:
/// - Each cached pattern stores its complete FSM structure
/// - Memory per pattern varies with pattern complexity
/// - Consider available RAM when configuring
///
/// # Performance Implications
/// - Larger cache → Better hit rate → Faster pattern matching
/// - Smaller cache → Lower memory usage → Possible repeated computation
pub static FSM_CACHE_SIZE: Lazy<usize> = Lazy::new(|| {
    env::var("FASTER_OUTLINES_CACHE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100)
});

/// Global flag to disable the FSM caching system.
///
/// # Environment Configuration
/// Set via `FASTER_OUTLINES_DISABLE_CACHE` environment variable.
///
/// # Accepted Values
/// - Enable cache (default):
///   - Not set
///   - Empty string
///   - "0"
///   - "false"
///   - "no"
/// - Disable cache:
///   - "1"
///   - "true"
///   - "yes"
///   Case insensitive for all values
///
/// control the env var like so:
/// ```bash
/// # Disable caching
/// export FASTER_OUTLINES_DISABLE_CACHE=true
///
/// # Enable caching (default)
/// export FASTER_OUTLINES_DISABLE_CACHE=false
/// # or
/// unset FASTER_OUTLINES_DISABLE_CACHE
/// ```
///
/// # When to Disable
/// 1. Debugging FSM computation
/// 2. Testing pattern matching
/// 3. Memory-critical environments
/// 4. Ensuring deterministic behavior
///
/// # Logging Behavior
/// Prints confirmation message to stdout when cache is disabled
pub static DISABLE_CACHE: Lazy<bool> =
    Lazy::new(|| match env::var("FASTER_OUTLINES_DISABLE_CACHE") {
        Ok(val) => {
            let val_lower = val.to_lowercase();
            let is_disabled = val_lower == "1" || val_lower == "true" || val_lower == "yes";
            if is_disabled {
                println!("Cache is disabled via DISABLE_CACHE environment variable.");
            }
            is_disabled
        }
        Err(_) => false,
    });
