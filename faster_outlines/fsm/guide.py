import interegular
from typing import Optional, List
from dataclasses import dataclass
from .regex import (
    create_fsm_index_end_to_end,
    fsm_to_betterfsm,
    create_fsm_index_tokenizer,
    build_regex
)
from .fsm_utils import TokenVocabulary
import torch

@dataclass(frozen=True)
class Write:
    """Write instruction.

    Attributes
    ----------
    tokens
        The sequence of tokens to be added to the current sequence by the
        generation process.

    """

    tokens: List[int]

@dataclass(frozen=True)
class Generate:
    """Generate instruction

    Attributes
    ----------
    tokens
        The tokens that lead to a valid completion if generated.  A value
        of ``None`` indicates that all tokens are allowed.
    """

    tokens: Optional[List[int]]

class RegexGuide():
    """Guide to generate text in the language of a regular expression."""

    def __init__(self, regex_string: str, tokenizer: TokenVocabulary):
        (
            self.fsm,
            fsm_finals,
            self.states
        ) = create_fsm_index_end_to_end(regex_string, tokenizer)
        self.eos_token_id = tokenizer.eos_token_id
        self.final_states = fsm_finals | {-1}
        self.states_to_token_maps = {}
    
    @property
    def states_to_token_maps(self):
        self.fsm.get_states_to_token_subsets()
    
    def get_next_instruction(self, state: int):
        """Return the next instruction for guided generation.
        In this function, we do not do any computation awaiting,
        since you must have called get_next_state before this function,
        and thus the states data should have already been stored
        """
        next_tokens_mask = self.states_to_token_masks.get(state)
        if next_tokens_mask is None:
            return Write(torch.tensor([self.eos_token_id]))

        return Generate(next_tokens_mask)

    def get_next_state(self, state: int, token_id: int) -> int:
        """Update the state of the guide."""
        if token_id == self.eos_token_id or state not in self.states:
            return -1

        state_map = self.states_to_token_maps[state]
        if state_map is None:
            # if the state was not going to be computed, we would not have made it this far.
            # wait for the state to finish computation. 
            self._fetch_states(wait_for=state)
            state_map = self.states_to_token_maps[state]
            assert state_map is not None
        next_state = state_map.get(token_id)
        if next_state is None:
            return -1
            
        return next_state


    def _fetch_states(self, wait_for = None):
        if wait_for is not None:
            self.fsm.await_state(wait_for)
        m, self.is_compilation_finished = self.fsm.collect_finished_states()
        self.states_to_token_maps.update(m)
        self.states_to_token_masks.update({
            k: torch.tensor(list(mapping.keys()), dtype=torch.int)
            for k, mapping in m.items()
        })
        
    @classmethod
    def from_interegular_fsm(
        cls, interegular_fsm: interegular.fsm.FSM, tokenizer
    ):
        instance = cls.__new__(cls)

        fsm = interegular_fsm.reduce()
        fsm = fsm_to_betterfsm(fsm)

        (
            instance.fsm,
        ) = create_fsm_index_tokenizer(fsm, tokenizer)

        instance.eos_token_id = tokenizer.eos_token_id
        return instance
    


class LazyVLLMRegexGuide():
    """Guide for generating text based on a regular expression.

    This implementation is designed specifically for VLLM, and defers computation
    until after the guide has been unpickled.

    The deferred computation is necessary due to VLLM's architecture, which uses 
    multiple Python interpreter processes. If computation starts in the client 
    interpreter before the object is transferred to the engine interpreter, the guide 
    will become disconnected from the underlying cache and the associated Rust Guide 
    object, since the cache is tied to the lifetime of the module.

    By delaying the computation until after the VLLMRegexGuide object is unpickled, 
    we ensure that it happens within the engine interpreter, where it's safe to access 
    the cache and complete the necessary operations.
    """

    def __init__(self, regex_string: str, tokenizer: TokenVocabulary):
        self.regex_string = regex_string
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.final_states = set()
        self.states_to_token_maps = {}
        self.states_to_token_masks = None
        self.fsm = None
        self.states = []
        # We compute the FSM before unpickling / entering the inference thread,
        # this way the inference thread is never blocked for large computations like fsm compilation
        # also because interegular.fsm.FSM is serializable, unlike the LazyFSMIndex.
        self.base_fsm = build_regex(regex_string)
        self.final_states = set(self.base_fsm.fsm_info['finals']) | {-1}
        self.states = self.base_fsm.fsm_info['states']

    def get_next_instruction(self, state: int):
        """Return the next instruction for guided generation.
        In this function, we do not do any computation awaiting,
        since you must have called get_next_state before this function,
        and thus the states data should have already been stored.
        """

        if state in self.final_states:
            return Write(torch.tensor([self.eos_token_id]))
        
        if state not in self.states_to_token_masks:
            self._fetch_states(wait_for=state)

        # Fetch the token mask for the current state
        next_tokens_mask = self.states_to_token_masks.get(state)
        
        if next_tokens_mask is None:
            return Write(torch.tensor([self.eos_token_id]))

        return Generate(next_tokens_mask)

    def get_next_state(self, state: int, token_id: int) -> int:
        """Update the state of the guide."""
        if state in self.final_states:
            return -1
            
        # Check if the token is the EOS token or the state doesn't exist
        if token_id == self.eos_token_id or state not in self.states:
            return -1

        # Fetch the state-to-token map for the current state
        state_map = self.states_to_token_maps.get(state)
        
        if state_map is None:
            # The state is being computed, wait for it to finish computation
            self._fetch_states(wait_for=state)
            state_map = self.states_to_token_maps.get(state)
            if state_map is None:
                raise RuntimeError(f"Failed to fetch state map for state: {state}")
        
        # Get the next state using the token_id
        next_state = state_map.get(token_id)
        if next_state is None:
            return -1

        return next_state

    def _fetch_states(self, wait_for = None):
        if wait_for is not None:
            self.fsm.await_state(wait_for)
        m = self.fsm.collect_finished_states()
        self.states_to_token_maps.update(m)
        self.states_to_token_masks.update({
            k: torch.tensor(list(mapping.keys()), dtype=torch.int)
            for k, mapping in m.items()
        })
        
    @classmethod
    def from_interegular_fsm(
        cls, interegular_fsm: interegular.fsm.FSM, tokenizer
    ):
        instance = cls.__new__(cls)

        fsm = interegular_fsm.reduce()
        fsm = fsm_to_betterfsm(fsm)

        (
            instance.fsm,
        ) = create_fsm_index_tokenizer(fsm, tokenizer)

        instance.eos_token_id = tokenizer.eos_token_id
        return instance

    def __getstate__(self):
        """Prepare the object for pickling. Do not include FSM in the state."""
        # Store the regex string and tokenizer in the pickle, but not the FSM
        # pickleing the fsm causes explosions.
        state = self.__dict__.copy()
        state["fsm"] = None
        return state

    def __setstate__(self, state):
        """Restore the object after unpickling."""
        self.__dict__.update(state)
        if self.regex_string and self.tokenizer:
            (
                self.fsm,
                self.empty_token_ids
            ) = create_fsm_index_tokenizer(
                self.base_fsm,
                self.tokenizer
            )
            self.states_to_token_masks = {}
            self._fetch_states(wait_for=1)
            

class VLLMRegexGuide():
    """Guide for generating text based on a regular expression.

    This implementation is designed specifically for VLLM, and defers computation
    until after the guide has been unpickled.

    The deferred computation is necessary due to VLLM's architecture, which uses 
    multiple Python interpreter processes. If computation starts in the client 
    interpreter before the object is transferred to the engine interpreter, the guide 
    will become disconnected from the underlying cache and the associated Rust Guide 
    object, since the cache is tied to the lifetime of the module.

    By delaying the computation until after the VLLMRegexGuide object is unpickled, 
    we ensure that it happens within the engine interpreter, where it's safe to access 
    the cache and complete the necessary operations.
    """

    def __init__(
        self,
        states_to_token_maps,
        fsm_finals,
        eos_token_id,
    ):
        self.eos_token_id = eos_token_id
        self.final_states = set(fsm_finals) | {-1}
        self.states_to_token_maps = states_to_token_maps
        self.states_to_token_masks = {
            state: torch.tensor(list(next_tokens_to_end_states.keys()))
            for state, next_tokens_to_end_states in states_to_token_maps.items()
        }

    def get_next_instruction(self, state: int):
        """Return the next instruction for guided generation.
        In this function, we do not do any computation awaiting,
        since you must have called get_next_state before this function,
        and thus the states data should have already been stored.
        """
        next_tokens_mask = self.states_to_token_masks.get(state)
        if next_tokens_mask is None:
            return Write(torch.tensor([self.eos_token_id]))

        return Generate(next_tokens_mask)

    def get_next_state(self, state: int, token_id: int) -> int:
        """Update the state of the guide."""
            
        if token_id == self.eos_token_id or state in self.final_states:
            return -1

        state_map = self.states_to_token_maps[state]

        next_state = state_map.get(token_id)
        if next_state is None:
            return -1
        
        return next_state
        
    @classmethod
    def from_interegular_fsm(
        cls, interegular_fsm: interegular.fsm.FSM, tokenizer
    ):
        instance = cls.__new__(cls)

        fsm = interegular_fsm.reduce()
        fsm = fsm_to_betterfsm(fsm)

        (
            instance.fsm,
            instance.empty_token_ids,
        ) = create_fsm_index_tokenizer(fsm, tokenizer)

        instance.eos_token_id = tokenizer.eos_token_id
        return instance
            
