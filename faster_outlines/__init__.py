# Copyright 2024 Nathan Hoos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .patch import patch
from .fsm import *

__all__ = [
  "FsmTokenizer", 
  "create_fsm_index_tokenizer", 
  "FSMState", 
  "patch",
  "Generate", 
  "Write"
]

__doc__ = """
faster_outlines: High-Performance Backend for Outlines

This module provides a Rust-powered backend replacement for the Outlines library,
offering significantly improved performance for structured text generation.

Usage:
------
To use the `faster_outlines` Rust backend with Outlines, import and apply the patch function:

    import outlines
    from faster_outlines import patch as patch_outlines

    patched_outlines = patch_outlines(outlines)
    # Or:
    patch_outlines(outlines)
    
    # Now just use outlines as usual.

Performance:
------------
This patch replaces the original Jitted Python (via Numba) backend with a Rust
implementation, providing substantial speed improvements, especially for
regex-structured generation and other computationally intensive tasks.

Note:
-----
- This patch modifies the Outlines module in-place. All existing and future
  references to the Outlines module will use the patched version.
- Apply this patch as early as possible in your application to ensure
  consistent behavior.
- While this patch aims for full compatibility with the Outlines API, 
  always test thoroughly after applying the patch to ensure your specific
  use case is not affected.

For more information and updates, visit:
https://github.com/unaidedelf8777/faster-outlines
"""