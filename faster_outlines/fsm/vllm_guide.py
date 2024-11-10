import interegular
from .regex import (
    create_fsm_index_tokenizer,
    build_regex,
    create_fsm_info
)
from faster_outlines.lib import TokenVocabulary

class LazyVLLMRegexGuide():
    """Guide for generating text based on a regular expression.

    This implementation is designed specifically for VLLM, and defers index computation
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
        self.fsm = None
        # We compute the FSM before unpickling / entering the inference thread,
        # this way the inference thread is never blocked for large computations like fsm compilation
        # also because interegular.fsm.FSM is serializable, unlike the LazyFSMIndex.
        self.base_fsm = build_regex(regex_string)
        
    @classmethod
    def from_interegular_fsm(
        cls, interegular_fsm: interegular.FSM, tokenizer
    ):
        instance = cls.__new__(cls)
        fsm = interegular_fsm.reduce()
        fsm = create_fsm_info(fsm)
        instance.fsm = create_fsm_index_tokenizer(fsm, tokenizer)
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
            
            self.fsm = create_fsm_index_tokenizer(
                self.base_fsm,
                self.tokenizer
            )
            self.get_next_state = self.fsm.get_next_state
            self.get_next_instruction = self.fsm.get_next_instruction
            del self.base_fsm
            del self.tokenizer