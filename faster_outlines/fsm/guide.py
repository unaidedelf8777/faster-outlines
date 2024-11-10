import interegular
from .regex import (
    create_fsm_index_end_to_end,
    create_fsm_index_tokenizer,
    create_fsm_info
)
from faster_outlines.lib import TokenVocabulary

class RegexGuide():
    """Guide to generate text in the language of a regular expression."""

    def __init__(self, regex_string: str, tokenizer: TokenVocabulary):
        (
            self.fsm,
            _
        ) = create_fsm_index_end_to_end(regex_string, tokenizer)
        self.eos_token_id = tokenizer.eos_token_id
        self.get_next_state = self.fsm.get_next_state
        self.get_next_instruction = self.fsm.get_next_instruction

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