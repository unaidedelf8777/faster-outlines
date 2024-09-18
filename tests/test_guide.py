import pytest
from faster_outlines import patch
import outlines
patch(outlines)
from outlines.fsm.guide import Generate, RegexGuide, StopAtEOSGuide, Write


def assert_expected_tensor_ids(tensor, ids):
    assert len(tensor) == len(ids)
    norm_tensor = sorted(map(int, tensor))
    norm_ids = sorted(map(int, tensor))
    assert norm_tensor == norm_ids, (norm_tensor, norm_ids)


def test_stop_at_eos():
    class MockTokenizer:
        vocabulary = {"a": 1, "eos": 2}
        eos_token_id = 2

    fsm = StopAtEOSGuide(MockTokenizer())

    instruction = fsm.get_next_instruction(fsm.start_state)
    assert isinstance(instruction, Generate)
    assert instruction.tokens is None

    instruction = fsm.get_next_instruction(fsm.final_state)
    assert isinstance(instruction, Write)
    assert instruction.tokens == [2]

    assert fsm.get_next_state(fsm.start_state, 2) == fsm.final_state
    assert fsm.get_next_state(fsm.start_state, 1) == fsm.start_state
    assert fsm.is_final_state(fsm.start_state) is False
    assert fsm.is_final_state(fsm.final_state) is True


#def test_regex_vocabulary_error():
#    class MockTokenizer:
#        vocabulary = {"a": 1}
#        special_tokens = {"eos"}
#        eos_token_id = 3
#
#        def convert_token_to_string(self, token):
#            return token
#
#    regex_str = "[1-9]"
#
#    with pytest.raises(ValueError, match="The vocabulary"):
#        RegexGuide(regex_str, MockTokenizer())
