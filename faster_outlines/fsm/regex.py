# Some of this files code is either partly or fully from the
# Official outlines repo, so credit to them.
# It's just included here because it makes things easier to optimize.
from typing import Dict, Tuple

from faster_outlines.fsm.fsm_utils import (
    create_fsm_index_end_to_end_rs,
    TokenVocabulary,
    # pywrite can't find the class for some reason.
    FSMInfo, #type: ignore
)
from interegular.fsm import FSM, Alphabet, anything_else
from interegular import parse_pattern
from functools import lru_cache


class BetterAlphabet(Alphabet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert anything_else in self._symbol_mapping
        self.anything_value = self._symbol_mapping[anything_else]

    def __getitem__(self, item):
        return self._symbol_mapping.get(item, self.anything_value)

    def copy(self):
        return BetterAlphabet(self._symbol_mapping.copy())


class BetterFSM(FSM):
    flat_transition_map: Dict[Tuple[int, int], int]

    def __init__(self, pattern: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.alphabet, BetterAlphabet):
            self.__dict__["alphabet"] = BetterAlphabet(self.alphabet._symbol_mapping)

        flat_transition_map = {}
        for from_state, trans_map in self.map.items():
            for trans_key, to_state in trans_map.items():
                flat_transition_map[(from_state, trans_key)] = to_state

        self.__dict__["flat_transition_map"] = flat_transition_map
        self.__dict__["_fsm_info"] = None
        if pattern is not None:
            self.__dict__["pattern"] = pattern
        else:
            self.__dict__["pattern"] = str(self.__hash__())

    @property
    def fsm_info(self):
        if self.__dict__["_fsm_info"] is None:
            anything_value = self.alphabet.anything_value #type: ignore
            self.__dict__["_fsm_info"] = {
                "initial": self.initial,
                "finals": list(self.finals),
                "transitions": self.flat_transition_map,
                "alphabet_anything_value": anything_value,
                "alphabet_symbol_mapping": {
                    k: v
                    for k, v in self.alphabet._symbol_mapping.items()
                    if k != anything_else
                },
                "pattern": self.pattern, #type: ignore
            }

        return self.__dict__["_fsm_info"]

    def __hash__(self):
        return hash(
            (
                frozenset(self.flat_transition_map.items()),
                frozenset(self.alphabet._symbol_mapping.items()),
            )
        )


def fsm_to_betterfsm(fsm: FSM, pattern=None):
    return BetterFSM(
        pattern=pattern, #type: ignore
        alphabet=BetterAlphabet(fsm.alphabet._symbol_mapping),
        initial=0,
        finals=fsm.finals,
        map=fsm.map,
        states=fsm.map.keys(),
    )

@lru_cache()
def build_regex(regex_string):
    return fsm_to_betterfsm(parse_pattern(regex_string).to_fsm().reduce(), regex_string)


def create_fsm_index_end_to_end(
    regex_str: str,
    vocabulary: TokenVocabulary,
):
    """Construct an FSM index from a tokenizer.

    This uses the end-to-end approach of `create_fsm_index_end_to_end`.
    """
    fsm = build_regex(regex_str)
    fsm_info = FSMInfo(**fsm.fsm_info)
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
