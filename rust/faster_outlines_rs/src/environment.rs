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
        .unwrap_or(50)
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
