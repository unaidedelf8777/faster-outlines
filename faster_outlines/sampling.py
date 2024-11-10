# Credit: This is the same logits processor from 
# VLLM. specifically from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/guided_decoding/outlines_logits_processors.py
# Adapted to be more lightweight, and drop support for the CFGGuide used in the file.
# Licensed under the Apache-2.0.
# if needed, obtain a copy of the license from https://www.apache.org/licenses/LICENSE-2.0
import torch
from typing import List
from collections import defaultdict
from faster_outlines.fsm import Write, Generate

class BaseLogitsProcessor:
    def __init__(self, guide):
        self._guide = guide
        self._fsm_state = defaultdict(int)

    def __call__(self, input_ids: List[int],
                 scores: torch.Tensor) -> torch.Tensor:
        """Use the FSM to bias the logits before sampling the next token."""
        seq_id = hash(tuple(input_ids))

        if len(input_ids) > 0:
            last_token = input_ids[-1]
            last_seq_id = hash(tuple(input_ids[:-1]))
            self._fsm_state[seq_id] = self._guide.get_next_state(
                state=self._fsm_state[last_seq_id], token_id=last_token)

        instruction = self._guide.get_next_instruction(
            state=self._fsm_state[seq_id])
        if type(instruction) == Generate:  # noqa: E721
            allowed_tokens = instruction.tokens
        elif type(instruction) == Write:  # noqa: E721
            allowed_tokens = [instruction.tokens[0]]
        else:
            raise TypeError(
                f"Unsupported instruction type {type(instruction)}")

        mask = torch.full((scores.shape[-1], ),
                          float("-inf"),
                          device=scores.device)
        mask[allowed_tokens] = 0
        scores.add_(mask)
        return scores