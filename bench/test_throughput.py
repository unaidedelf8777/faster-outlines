import time
import random
from faster_outlines.fsm import FsmTokenizer
from transformers import AutoTokenizer
from faster_outlines.fsm.regex import create_fsm_index_end_to_end

test_patterns = [
    r"""\{[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.){,10}"[\n ]*,[\n ]*"age"[\n ]*:[\n ]*(0|[1-9][0-9]*)[\n ]*,[\n ]*"armor"[\n ]*:[\n ]*("leather"|"chainmail"|"plate")[\n ]*,[\n ]*"strength"[\n ]*:[\n ]*(0|[1-9][0-9]*)[\n ]*\}""",
    r"""\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"work"[\n ]*:[\n ]*\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"composer"[\n ]*:[\n ]*\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"functions"[\n ]*:[\n ]*\[("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*")(,("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"))*\][\n ]*\}[\n ]*\}[\n ]*,[\n ]*"recording_artists"[\n ]*:[\n ]*\[(\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"functions"[\n ]*:[\n ]*\[("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*")(,("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"))*\][\n ]*\})(,(\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"functions"[\n ]*:[\n ]*\[("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*")(,("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"))*\][\n ]*\}))*\][\n ]*\}""",
    r"""\{[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.){,10}"[\n ]*,[\n ]*"age"[\n ]*:[\n ]*(0|[1-9][0-9]*)[\n ]*,[\n ]*"armor"[\n ]*:[\n ]*("leather"|"chainmail"|"plate")[\n ]*,[\n ]*"strength"[\n ]*:[\n ]*(0|[1-9][0-9]*)[\n ]*\}""",
    r"""\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"work"[\n ]*:[\n ]*\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"composer"[\n ]*:[\n ]*\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"functions"[\n ]*:[\n ]*\[("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*")(,("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"))*\][\n ]*\}[\n ]*\}[\n ]*,[\n ]*"recording_artists"[\n ]*:[\n ]*\[(\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"functions"[\n ]*:[\n ]*\[("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*")(,("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"))*\][\n ]*\})(,(\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"functions"[\n ]*:[\n ]*\[("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*")(,("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"))*\][\n ]*\}))*\][\n ]*\}""",
]
# Initialize the tokenizer
tokenizer = FsmTokenizer(
    AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
)

def measure_throughput(fsm, N):
    current_state = 0  # Start state is always 0
    num_instructions = 0
    start_time = time.perf_counter()
    for _ in range(N):
        instruction = fsm.get_next_instruction(current_state)
        allowed_tokens = list(instruction.tokens)
        if not allowed_tokens:
            current_state = 0  # Reset to start state if no allowed tokens
            continue
        token_id = random.choice(allowed_tokens)
        next_state = fsm.get_next_state(current_state, token_id)
        current_state = next_state
        num_instructions += 1
    end_time = time.perf_counter()
    total_time = end_time - start_time
    instructions_per_second = num_instructions / total_time
    print(f"Total instructions: {num_instructions}")
    print(f"Total time: {total_time:.4f} seconds")
    return instructions_per_second, total_time

def test_throughput():
    for i, pattern in enumerate(test_patterns):
        print(f"\nTesting pattern {i + 1}/{len(test_patterns)}")
        print(f"Pattern: {pattern[:50]}...")  # Print first 50 chars of pattern

        fsm, empty_token_ids, fsm_finals = create_fsm_index_end_to_end(pattern, tokenizer)
        fsm.get_states_to_token_subsets()
        N = 1000000  # Number of instructions to simulate
        instructions_per_second, total_time = measure_throughput(fsm, N)
        print(f"Instructions per second: {instructions_per_second:.2f}")
        time_per_instruction = (total_time / N) * 1e6
        print(f"Time per instruction: {time_per_instruction:.3f} µs")
        print("====================================")

# Run the throughput test
test_throughput()
