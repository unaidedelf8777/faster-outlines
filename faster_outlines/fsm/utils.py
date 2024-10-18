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

from functools import lru_cache


@lru_cache
def reduced_vocabulary(tokenizer):
    """Create a map from decoded vocabulary tokens to lists of equivalent token ids."""
    vocabulary = {}
    empty_token_ids = set()
    for token, token_idx in tokenizer.vocabulary.items():
        if token in tokenizer.special_tokens:
            continue

        token_str = tokenizer.convert_token_to_string(token)

        if token_str:
            # Ensure the key exists with an empty list, then append
            vocabulary.setdefault(token_str, []).append(token_idx)
        else:
            empty_token_ids.add(token_idx)

    return vocabulary, empty_token_ids
