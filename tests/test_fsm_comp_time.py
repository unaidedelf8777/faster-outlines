import time
import json
import os
import faster_outlines
from faster_outlines.fsm import FsmTokenizer
from transformers import AutoTokenizer
from json import dumps as json_dumps

from outlines.fsm.guide import RegexGuide
from outlines import clear_cache, disable_cache

clear_cache()
disable_cache()

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
    r"""choice 1|choice 2|car|truck|dog""",
    r"""long choice 1 giberrish blah blah blah|long choice 2 giberrish blah blah blah""",
    r"""Despite the serene landscape, he felt a profound sense of urgency as the clouds began to gather ominously overhead.|She meticulously arranged the old books in alphabetical order, each dusty spine telling a story of forgotten worlds."""
    
]
tokenizer = FsmTokenizer(
    AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
)

def load_previous_results():
    if os.path.exists('benchmark_results.json'):
        with open('benchmark_results.json', 'r') as f:
            return json.load(f)
    return {}

def save_results(results):
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

def test_benchmark_compile_fsm():
    from faster_outlines.fsm.regex import create_fsm_index_tokenizer

    previous_results = load_previous_results()
    current_results = {}

    # let it run once so it imports everything it needs
    # and gets primed, just to be fair.
    RegexGuide(test_patterns[0], tokenizer)
    
    
    for i, pattern in enumerate(test_patterns):
        
        st = time.perf_counter()
        rfsm = RegexGuide(pattern, tokenizer)
        
        outlines_time = time.perf_counter() - st
        
        print(f"Time taken by `Outlines`: {outlines_time}")
        
        total_time = 0
        iterations = 6
        pattern_times = []

        print(f"Testing pattern {i + 1}/{len(test_patterns)}")
        print(f"Pattern: {pattern[:50]}...")  # Print first 50 chars of pattern
        return_time = None
        for j in range(iterations):
            start_time = time.perf_counter()
            fsm, empty_token_ids, fsm_finals = create_fsm_index_tokenizer(pattern, tokenizer)
            fsm.get_next_instruction(0)
            time_to_return = time.perf_counter()
            fsm.get_states_to_token_subsets()
            end_time = time.perf_counter()
            computation_time = end_time - start_time
            total_time += computation_time
            pattern_times.append(computation_time)
            
            if return_time is None:
                return_time = time_to_return - start_time
                print(f"Return time: {return_time}")
                print(f"initial tokens: {[tokenizer.decode([x])[0] for x in fsm.allowed_token_ids(0)[:10]]}")

            print(f"Iteration {j + 1}: {computation_time:.4f} seconds")

        average_time = total_time / iterations
        current_results[pattern] = {
            'average_time': average_time,
            'outlines_time': outlines_time,
            'times': pattern_times,
            'states_length': len(rfsm.states_to_token_maps.keys()),
            'return_time': return_time
        }

        print(f"Average time: {average_time:.4f} seconds")
        print(f"Win / Loss?: {'Win' if average_time < outlines_time else 'Loss'}")

        if pattern in previous_results:
            prev_avg = previous_results[pattern]['average_time']
            improvement = (prev_avg - average_time) / prev_avg * 100
            print(f"Performance change: {improvement:.2f}% {'improvement' if improvement > 0 else 'deterioration'}")
        else:
            print("No previous data for comparison")

        print("====================================")

    save_results(current_results)
    return current_results

def compare_results(current_results, previous_results):
    print("\nOverall Performance Comparison:")
    print("================================")
    for pattern, current_data in current_results.items():
        if pattern in previous_results:
            prev_avg = previous_results[pattern]['average_time']
            curr_avg = current_data['average_time']
            improvement = (prev_avg - curr_avg) / prev_avg * 100
            print(f"Pattern: {pattern[:30]}...")
            print(f"  Previous average: {prev_avg:.4f} seconds")
            print(f"  Current average:  {curr_avg:.4f} seconds")
            print(f"  Change: {improvement:.2f}% {'improvement' if improvement > 0 else 'deterioration'}")
        else:
            print(f"Pattern: {pattern[:30]}... (New pattern, no previous data)")
        print("------------------------------------")

# Run the benchmark test
previous_results = load_previous_results()
current_results = test_benchmark_compile_fsm()
compare_results(current_results, previous_results)