from typing import Dict

from faster_outlines.lib import (
    create_fsm_index_end_to_end_rs,
    TokenVocabulary,
    # pywrite can't find the class for some reason.
    FSMInfo, #type: ignore
)
from interegular.fsm import FSM, anything_else
from interegular import parse_pattern
from functools import lru_cache


def create_fsm_info(
    fsm: FSM, 
    pattern: str = None #type: ignore
) -> dict:
    """
    Creates FSM info dictionary with flattened transition map and alphabet information.
    Replaces functionality previously in BetterFSM and BetterAlphabet classes.
    """
    flat_transition_map = {}
    for from_state, trans_map in fsm.map.items():
        for trans_key, to_state in trans_map.items():
            flat_transition_map[(from_state, trans_key)] = to_state
            
    anything_value = fsm.alphabet._symbol_mapping[anything_else]
    
    return {
        "initial": fsm.initial,
        "finals": list(fsm.finals),
        "transitions": flat_transition_map,
        "alphabet_anything_value": anything_value,
        "alphabet_symbol_mapping": {
            k: v
            for k, v in fsm.alphabet._symbol_mapping.items()
            if k != anything_else
        },
        "pattern": pattern if pattern is not None else str(sorted([
            (from_state, symbol, to_state) 
            for (from_state, symbol), to_state in flat_transition_map.items()
        ])) + "_" + str(sorted(
            [(k, v) for k, v in fsm.alphabet._symbol_mapping.items() if k != anything_else]
        )),
    }

@lru_cache()
def build_regex(regex_string: str) -> Dict:
    """
    Builds a regex FSM and returns its info dictionary.
    """
    fsm = parse_pattern(regex_string).to_fsm().reduce()
    return create_fsm_info(fsm, regex_string)

def create_fsm_index_end_to_end(
    regex_str: str,
    vocabulary: TokenVocabulary,
):
    """Construct an FSM index from a tokenizer.

    This uses the end-to-end approach of `create_fsm_index_end_to_end`.
    """
    fsm = build_regex(regex_str)
    fsm_info = FSMInfo(**fsm)
    finals = set(fsm_info.finals)
    lazy_fsm_index = create_fsm_index_end_to_end_rs(fsm_info, vocabulary)
    return lazy_fsm_index, finals


def create_fsm_index_tokenizer(
    fsm_info: dict, vocabulary: TokenVocabulary, frozen_tokens=None
):
    """NOTE: frozen_tokens will be ignored. it is only there to align with the outlines api."""
    fsm_info = FSMInfo(**fsm_info)

    lazy_fsm_index = create_fsm_index_end_to_end_rs(fsm_info, vocabulary)

    return lazy_fsm_index
