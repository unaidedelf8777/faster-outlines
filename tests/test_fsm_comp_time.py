import time
import faster_outlines
from faster_outlines.fsm import FsmTokenizer
from transformers import AutoTokenizer
from json import dumps as json_dumps

test_patterns = [
    r"""[a-z0-9!#$%&'*+/=?^_{|}~-]+(?:.[a-z0-9!#$%&'*+/=?^_{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?""",
    r"""[a-z0-9!#$%&'*+/=?^_{|}~-]+(?:.[a-z0-9!#$%&'*+/=?^_{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?""",
    r"""\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}""",
    r"""\+?[1-9][0-9]{7,14}""",
    r"""([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])|([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])""",
    r"""(0?[1-9]|1[0-2]):[0-5]\d\s?(am|pm)?""",
    r"""(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)""",
    r"""(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?""",
    r"""\d{3}-\d{2}-\d{4}""",
    r"""\{[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.){,10}"[\n ]*,[\n ]*"age"[\n ]*:[\n ]*(0|[1-9][0-9]*)[\n ]*,[\n ]*"armor"[\n ]*:[\n ]*("leather"|"chainmail"|"plate")[\n ]*,[\n ]*"strength"[\n ]*:[\n ]*(0|[1-9][0-9]*)[\n ]*\}""",
    r"""\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"work"[\n ]*:[\n ]*\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"composer"[\n ]*:[\n ]*\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"functions"[\n ]*:[\n ]*\[("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*")(,("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"))*\][\n ]*\}[\n ]*\}[\n ]*,[\n ]*"recording_artists"[\n ]*:[\n ]*\[(\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"functions"[\n ]*:[\n ]*\[("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*")(,("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"))*\][\n ]*\})(,(\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"functions"[\n ]*:[\n ]*\[("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*")(,("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"))*\][\n ]*\}))*\][\n ]*\}""",
]

tokenizer = FsmTokenizer(
        AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
    )
# A function to perform the benchmark test
def test_benchmark_compile_fsm():

    from faster_outlines.fsm.regex import create_fsm_index_tokenizer

    # Benchmark phase
    for pattern in test_patterns:
        total_time = 0
        iterations = 6  # Set a constant number of iterations
        for i in range(1, iterations + 1):
            
            print("starting timer")
            start_time = time.perf_counter()
            fsm, empty_token_ids, fsm_finals = create_fsm_index_tokenizer(pattern, tokenizer)
            print(len(fsm.get_next_instruction(0).tokens))
            r = fsm.get_states_to_token_subsets()
            end_time = time.perf_counter()
            computation_time = end_time - start_time
            total_time += computation_time
            
            
            print(f"Time taken for Rust: {computation_time} seconds")
            print("====================================")
            print(f"first state: {0}")
            print(f"initial tokens: {[tokenizer.decode([x])[0] for x in fsm.allowed_token_ids(0)[:10]]}")
            print("====================================")
            time.sleep(0.5)

        average_time = total_time / iterations
        print("************************************")
        print(f"Average time for pattern '{pattern}': {average_time} seconds")
        print("************************************")
        


def test_patch_outlines_inplace():
    import outlines
    from faster_outlines import patch
    patched_outlines = patch(outlines)
    
    pattern = test_patterns[0]
    print(pattern)
    regex_fsm = patched_outlines.fsm.guide.RegexGuide(pattern, tokenizer)
    tokens = regex_fsm.get_next_instruction(2).tokens[:10]
    
    print(f"initial tokens: {[tokenizer.decode([x])[0] for x in list(tokens)[:10]]}")
    
    print()
    
    print("Patching test sucessful!")
# Run the benchmark test
test_patch_outlines_inplace()
test_benchmark_compile_fsm()
