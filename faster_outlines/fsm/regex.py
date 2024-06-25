from typing import TYPE_CHECKING, Dict, List, Set, Tuple, NewType

import numpy as np
from interegular.fsm import FSM, Alphabet, anything_else
from interegular import parse_pattern
from .fsm_utils import create_fsm_index_end_to_end
from .utils import reduced_vocabulary

if TYPE_CHECKING:
    from .tokenizer_fsm_patch import Tokenizer

FSMState = NewType("FSMState", int)

class BetterAlphabet(Alphabet):
    def __init__(self, *args, **kwargs):
        # type: (*Dict[str, int], **Any) -> None
        super().__init__(*args, **kwargs)
        assert anything_else in self._symbol_mapping
        self.anything_value = self._symbol_mapping[anything_else]

    def __getitem__(self, item):
        return self._symbol_mapping.get(item, self.anything_value)

    def copy(self):
        return BetterAlphabet(self._symbol_mapping.copy())


class BetterFSM(FSM):
    flat_transition_map: Dict[Tuple[int, int], int]
    trans_key_to_states: Dict[int, List[int]]

    def __init__(self, *args, **kwargs):
        # type: (*Any, **Any) -> None
        super().__init__(*args, **kwargs)

        if not isinstance(self.alphabet, BetterAlphabet):
            self.__dict__["alphabet"] = BetterAlphabet(self.alphabet._symbol_mapping)

        flat_transition_map = {}
        trans_key_to_states = {}
        for from_state, trans_map in self.map.items():
            for trans_key, to_state in trans_map.items():
                flat_transition_map[(from_state, trans_key)] = to_state
                trans_key_to_states.setdefault(trans_key, set()).add(from_state)

        self.__dict__["trans_key_to_states"] = trans_key_to_states
        self.__dict__["flat_transition_map"] = flat_transition_map
        self.__dict__["_fsm_info"] = None

    def copy(self):
        return BetterFSM(
            alphabet=self.alphabet.copy(),
            states=self.states.copy(),
            initial=self.initial,
            finals=self.finals.copy(),
            map=self.map.copy(),
            __no_validation__=True,
        )

    @property
    def fsm_info(self):
        if self._fsm_info is None:
            flat_transition_map_items = np.fromiter(
                ((a[0], a[1], b) for a, b in self.flat_transition_map.items()),
                dtype=np.dtype("i8, i8, i8"),
            )
            trans_key_to_states_items = np.fromiter(
                ((k, z) for k, v in self.trans_key_to_states.items() for z in v),
                dtype=np.dtype("i8, i8"),
            )
            alphabet_symbol_mapping_items = np.fromiter(
                (
                    it
                    for it in self.alphabet._symbol_mapping.items()
                    if it[0] != anything_else
                ),
                dtype=np.dtype("U1, i8"),
            )
            nb_finals = np.fromiter(self.finals, dtype=np.dtype("i8"))
            self.__dict__["_fsm_info"] = create_fsm_info(
                self.initial,
                nb_finals,
                flat_transition_map_items,
                trans_key_to_states_items,
                self.alphabet.anything_value,
                alphabet_symbol_mapping_items,
            )

        return self._fsm_info


def create_fsm_info(
    py_initial,
    py_finals,
    flat_transition_map_items,
    trans_key_to_states_items,
    py_anything_value,
    alphabet_symbol_mapping_items,
):
    trans_key_to_states = {}
    for trans_key, state in trans_key_to_states_items:
        if trans_key not in trans_key_to_states:
            trans_key_to_states[trans_key] = []
        trans_key_to_states[trans_key].append(state)

    flat_transition_map = {
        (from_state, trans_key): to_state
        for from_state, trans_key, to_state in flat_transition_map_items
    }

    alphabet_symbol_map = {
        symbol: trans_key for symbol, trans_key in alphabet_symbol_mapping_items
    }

    initial = py_initial
    finals = set(py_finals)
    anything_value = py_anything_value

    return {
        "initial": initial,
        "finals": finals,
        "transitions": flat_transition_map,
        "trans_key_to_states": trans_key_to_states,
        "alphabet_anything_value": anything_value,
        "alphabet_symbol_mapping": alphabet_symbol_map,
    }


def make_deterministic_fsm(fsm: FSM) -> Tuple[BetterFSM, Dict[int, int]]:
    """Construct an equivalent FSM with deterministic state labels."""
    old_to_new_trans_keys = {
        trans_key: i
        for i, (trans_key, _) in enumerate(
            sorted(fsm.alphabet.by_transition.items(), key=lambda x: sorted(x[1]))
        )
    }

    new_symbol_mapping = {
        symbol: old_to_new_trans_keys[trans_key]
        for symbol, trans_key in fsm.alphabet._symbol_mapping.items()
    }

    new_alphabet = BetterAlphabet(new_symbol_mapping)

    new_map = {
        from_state: {
            old_to_new_trans_keys[trans_key]: to_state
            for trans_key, to_state in trans_map.items()
        }
        for from_state, trans_map in fsm.map.items()
    }

    old_to_new_states = {}
    old_to_new_states[fsm.initial] = 0

    i = 0
    seen = {fsm.initial}
    old_state_queue = [fsm.initial]
    while old_state_queue:
        old_state = old_state_queue.pop(-1)
        transitions = new_map[old_state]
        sorted_transitions = sorted(transitions.items(), key=lambda v: v[0])
        for _, old_state in sorted_transitions:
            if old_state not in seen:
                old_state_queue.append(old_state)
                seen.add(old_state)
            if old_state not in old_to_new_states:
                i += 1
                old_to_new_states[old_state] = i

    new_map = dict(
        sorted(
            (
                (
                    old_to_new_states[from_state],
                    dict(
                        sorted(
                            (
                                (trans_key, old_to_new_states[to_state])
                                for trans_key, to_state in trans_map.items()
                            ),
                            key=lambda v: v[0],
                        )
                    ),
                )
                for from_state, trans_map in new_map.items()
            ),
            key=lambda v: v[0],
        )
    )

    new_initial = 0
    new_finals = frozenset(
        sorted(old_to_new_states[old_state] for old_state in fsm.finals)
    )
    new_states = frozenset(sorted(new_map.keys()))

    new_fsm = BetterFSM(new_alphabet, new_states, new_initial, new_finals, new_map)

    return new_fsm, old_to_new_states

def create_fsm_index_tokenizer(
    pattern: str,
    tokenizer: "Tokenizer",
) -> Tuple[Dict[int, Dict[int, int]], Set[int]]:
    """Construct an FMS index from a tokenizer.

    This uses the end-to-end approach of `create_fsm_index_end_to_end`.

    .. warning::

        `fsm` needs to be deterministically ordered so that future caching makes sense.

    """
    fsm, _ = make_deterministic_fsm(parse_pattern(pattern).to_fsm().reduce())
    vocabulary, empty_token_ids = reduced_vocabulary(tokenizer)

    fsm_info = fsm.fsm_info
    # rust impl expects generic types, so just cast them.
    lazy_fsm_index = create_fsm_index_end_to_end(fsm_info, dict(vocabulary), tokenizer.eos_token_id)  # type: ignore
    return lazy_fsm_index, empty_token_ids
