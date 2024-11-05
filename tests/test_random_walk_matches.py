# For ea

import pytest
import torch
import random
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer

from faster_outlines.fsm import RegexGuide, TokenVocabulary

@dataclass
class TestCase:
    pattern: str
    description: str

TEST_PATTERNS = [
    TestCase(r"[a-z]{1,10}", "simple lowercase word"),
    TestCase(r"[0-9]{3,5}", "3-5 digits"),
    TestCase(r"(foo|bar)(foo|bar)", "foo or bar repeated twice"),
    TestCase(r"[A-Z][a-z]{2,4}", "capitalized word 3-5 chars"),
    TestCase(r"(abc|ac){1,3}", "1-3 repetitions of abc or ac"),
    TestCase(r"[a-z0-9]+@[a-z0-9]+\.(com|org|net)", "simple email pattern"),
    TestCase(r"https?://[a-z0-9]+\.[a-z]{2,4}", "simple URL pattern"),
    TestCase(r"[aeiou]{2,5}", "2-5 vowels"),
    TestCase(r"[0-9]{2}-[0-9]{2}-[0-9]{4}", "date format xx-xx-xxxx"),
]

NUM_SAMPLES_PER_PATTERN = 50
MAX_SEQUENCE_LENGTH = 100  # Safety limit for generation

# we use a relatively small tokenizer just to speed things up.
t = AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B")

def create_mock_vocabulary() -> TokenVocabulary:
    """
    Create a mock vocabulary for testing.
    """
    return TokenVocabulary(
        t.get_vocab(),
        t.eos_token_id,
        set(t.all_special_tokens)
    )

def generate_sequence(guide, max_length: int = MAX_SEQUENCE_LENGTH) -> Tuple[List[int], Optional[str]]:
    """Generate a sequence of tokens following the guide's instructions."""
    tokens = []
    state = 0  
    
    while True:
        instruction = guide.get_next_instruction(state)
        if len(tokens) == max_length:
            return tokens, "max_length"
        
        if hasattr(instruction, 'tokens') and instruction.tokens is not None:
            if isinstance(instruction.tokens, torch.Tensor):
                allowed_tokens = instruction.tokens.tolist()
            else:
                allowed_tokens = instruction.tokens
                
            if len(allowed_tokens) == 1 and allowed_tokens[0] == guide.eos_token_id:
                tokens.append(guide.eos_token_id)
                break
                
            token_id = random.choice(allowed_tokens)
            tokens.append(token_id)
            
            state = guide.get_next_state(state, token_id)
            
            if state == -1:  
                if tokens[-1] != guide.eos_token_id:
                    tokens.append(guide.eos_token_id)
                break
    
    return tokens, None

def verify_sequence(pattern: str, sequence: str) -> bool:
    """Verify if the generated sequence matches the regex pattern using Python's re module."""
    try:
        return re.fullmatch(pattern, sequence) is not None
    except re.error:
        print(f"Invalid regex pattern: {pattern}")
        return False

@pytest.fixture(params=[RegexGuide]) 
def guide_class(request):
    """Parametrized fixture that provides each guide class."""
    return request.param

@pytest.fixture(scope="session")
def tokenizer():
    """Fixture for the tokenizer."""
    return AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")

@pytest.fixture(scope="session")
def vocab(tokenizer):
    """Fixture for the TokenVocabulary."""
    return TokenVocabulary(
        tokenizer.get_vocab(),
        tokenizer.eos_token_id,
        set(tokenizer.all_special_tokens)
    )

@pytest.mark.parametrize("test_case", TEST_PATTERNS)
def test_regex_guide(guide_class, vocab, tokenizer, test_case):
    """Test regex guide by random walking the FSM for each pattern.
    
    For each pattern, attempts NUM_SAMPLES_PER_PATTERN walks through the FSM,
    randomly choosing from allowed tokens at each state. Some walks may
    exceed maximum length and timeout - these are retried and don't count
    as failures. Only completed walks that don't match the pattern count
    as actual failures.
    """
    guide = guide_class(test_case.pattern, vocab)
    
    success_count = 0
    attempt_count = 0
    failed_sequences = []
    timeout_count = 0
    
    sample_idx = 0
    while sample_idx < NUM_SAMPLES_PER_PATTERN:
        try:
            token_sequence, code = generate_sequence(guide)

            if code is not None:
                timeout_count += 1
                continue  
            
            attempt_count += 1 
            
            decoded_sequence = tokenizer.decode(token_sequence[:-1])  
            
            is_valid = verify_sequence(test_case.pattern, decoded_sequence)
            
            if is_valid:
                success_count += 1
                if sample_idx < 3: 
                    print(f"✓ Valid sequence generated: {decoded_sequence}")
            else:
                if len(failed_sequences) < 3:  
                    failed_sequences.append(decoded_sequence)
            
            sample_idx += 1 
                
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            sample_idx += 1 
            continue
    
    for seq in failed_sequences:
        print(f"✗ Invalid sequence generated: {seq}")
    
    if attempt_count == 0:
        pytest.fail(f"All {timeout_count} attempts timed out, no successful generations")
    
    success_rate = (success_count / attempt_count) * 100
    timeout_rate = (timeout_count / (timeout_count + attempt_count)) * 100
    
    print(f"Total attempts: {timeout_count + attempt_count}")
    print(f"Timeout rate: {timeout_rate:.2f}% ({timeout_count} timeouts)")
    print(f"Pattern success rate: {success_rate:.2f}% (of {attempt_count} completed attempts)")
    
    assert success_rate == 100, \
        f"Success rate {success_rate:.2f}% below required 100% for completed sequences"