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

pub static FSM_CACHE_SIZE: Lazy<usize> = Lazy::new(|| {
    env::var("FASTER_OUTLINES_CACHE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100)
});

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
