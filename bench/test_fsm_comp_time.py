import time
import json
import os

os.environ['FASTER_OUTLINES_DISABLE_CACHE'] = 'true'

import faster_outlines
from faster_outlines.fsm import create_token_vocabulary_from_tokenizer, Write, Generate
from transformers import AutoTokenizer
from json import dumps as json_dumps

from outlines_core.fsm.guide import RegexGuide
from outlines import clear_cache, disable_cache
import torch
from typing import Union, List, Tuple


SPIECE_UNDERLINE = "\u2581"


class TransformerTokenizer():
    """Represents a tokenizer for models in the `transformers` library."""

    def __init__(self, tokenizer: "PreTrainedTokenizer", **kwargs):
        self.tokenizer = tokenizer
        self.hash = None
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.pad_token = self.tokenizer.pad_token

        self.special_tokens = set(self.tokenizer.all_special_tokens)

        self.vocabulary = self.tokenizer.get_vocab()

    def encode(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        kwargs["padding"] = True
        kwargs["return_tensors"] = "pt"
        output = self.tokenizer(prompt, **kwargs)
        return output["input_ids"], output["attention_mask"]

    def decode(self, token_ids: torch.LongTensor) -> List[str]:
        text = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return text

    def convert_token_to_string(self, token: str) -> str:
        string = self.tokenizer.convert_tokens_to_string([token])

        # A hack to handle missing spaces to HF's Llama tokenizers
        if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
            return " " + string

        return string

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if hasattr(self, "model_name") and hasattr(self, "kwargs"):
                return (
                    other.model_name == self.model_name and other.kwargs == self.kwargs
                )
            else:
                return other.tokenizer == self.tokenizer
        return NotImplemented

    def __hash__(self):
        from datasets.fingerprint import Hasher
        if self.hash is not None:
            return self.hash
        
        else:
            self.hash =  hash(Hasher.hash(self.tokenizer))
            return self.hash

clear_cache()
disable_cache()

test_patterns = [
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
]
# The difference in speed for both these tokenizers is roughly proportional
# to the number of tokens, so just bench with the smaller one.
tokenizer = TransformerTokenizer(
    AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
    # AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct") 
)

tokenvocab = create_token_vocabulary_from_tokenizer(
    AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
)

print(f"Benchmarking tokenizer length: {len(tokenizer.tokenizer)}")

def load_previous_results():
    if os.path.exists('bench/benchmark_results.json'):
        with open('bench/benchmark_results.json', 'r') as f:
            return json.load(f)
    return {}

def save_results(results):
    with open('bench/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

def test_benchmark_compile_fsm():
    from faster_outlines.fsm.regex import create_fsm_index_end_to_end

    previous_results = load_previous_results()
    current_results = {}

    RegexGuide.from_regex(test_patterns[0], tokenizer)  # Priming

    for i, pattern in enumerate(test_patterns):
        
        st = time.perf_counter()
        rfsm = RegexGuide.from_regex(pattern, tokenizer) 
        
        outlines_time = time.perf_counter() - st
        
        print(f"Time taken by `Outlines`: {outlines_time}")
        
        total_time = 0
        iterations = 10
        pattern_times = []

        print(f"Testing pattern {i + 1}/{len(test_patterns)}")
        print(f"Pattern: {pattern[:50]}...")  
        return_time = None
        for j in range(iterations):
            start_time = time.perf_counter()
            fsm, fsm_finals, _ = create_fsm_index_end_to_end(pattern, tokenvocab)
            time_to_return = time.perf_counter()
            fsm.await_finished()
            l = fsm.get_next_instruction(state=0)
            end_time = time.perf_counter()
            computation_time = end_time - start_time
            total_time += computation_time
            pattern_times.append(computation_time)
            
            if return_time is None:
                return_time = time_to_return - start_time
                print(f"Return time: {return_time}")
                print(f"Initial tokens: {[tokenizer.decode([x])[0] for x in fsm.get_allowed_token_ids(0)[:10]]}")

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
        print(f"Number of states: { len(rfsm.states_to_token_maps.keys())}")
        print(f"States processed per second: { len(rfsm.states_to_token_maps.keys()) / average_time:.2f}")
        
        # Win / Loss calculation
        if average_time < outlines_time:
            print("Result: Win")
            percentage = ((outlines_time - average_time) / outlines_time) * 100
            print(f"Percentage faster than `Outlines`: {percentage:.2f}%")
            print(f"{outlines_time / average_time}x faster than outlines")
        else:
            print("Result: Loss")
            percentage = ((average_time - outlines_time) / outlines_time) * 100
            print(f"Percentage slower than `Outlines`: {percentage:.2f}%")
            print(f"{outlines_time / average_time}x slower than outlines")

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
    total_our_time = 0
    total_outlines_time = 0

    for pattern, current_data in current_results.items():
        curr_avg = current_data['average_time']
        outlines_time = current_data['outlines_time']
        total_our_time += curr_avg
        total_outlines_time += outlines_time

        faster_than_outlines = (outlines_time - curr_avg) / outlines_time * 100

        print(f"Pattern: {pattern[:30]}...")
        print(f"  Current average:  {curr_avg:.4f} seconds")
        print(f"  Outlines time:    {outlines_time:.4f} seconds")
        if curr_avg < outlines_time:
            print(f"  Faster than Outlines by {faster_than_outlines:.2f}%")
        else:
            print(f"  Slower than Outlines by {-faster_than_outlines:.2f}%")
        
        if pattern in previous_results:
            prev_avg = previous_results[pattern]['average_time']
            improvement = (prev_avg - curr_avg) / prev_avg * 100
            print(f"  Previous average: {prev_avg:.4f} seconds")
            print(f"  Change since last run: {improvement:.2f}% {'improvement' if improvement > 0 else 'deterioration'}")
        else:
            print("  No previous data for comparison")
        print("------------------------------------")

    overall_faster_than_outlines = (total_outlines_time - total_our_time) / total_outlines_time * 100
    print("\nOverall Summary:")
    print("================================")
    print(f"Total time for our code: {total_our_time:.4f} seconds")
    print(f"Total time for Outlines: {total_outlines_time:.4f} seconds")
    if total_our_time < total_outlines_time:
        print(f"Our code is faster than Outlines by {overall_faster_than_outlines:.2f}% overall")
    else:
        print(f"Our code is slower than Outlines by {-overall_faster_than_outlines:.2f}% overall")
    print("================================")

# Run the benchmark test
previous_results = load_previous_results()
current_results = test_benchmark_compile_fsm()
compare_results(current_results, previous_results)
