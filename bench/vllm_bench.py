import asyncio
from openai import AsyncOpenAI, OpenAI
import time
import json
from datetime import datetime
import statistics
from typing import List, Dict, Any
from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")

prime_pattern = "\n\n"
REGEX_PATTERNS = [
    (r"""\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}""",47),
    (r"""\+?[1-9][0-9]{7,14}""", 16),
    (r"""([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])|([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])""", 34),
    (r"""(0?[1-9]|1[0-2]):[0-5]\d\s?(am|pm)?""", 9),
    (r"""(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)""", 23),
    (r"""(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?""", 13),
    (r"""\{[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.){,10}"[\n ]*,[\n ]*"age"[\n ]*:[\n ]*(0|[1-9][0-9]*)[\n ]*,[\n ]*"armor"[\n ]*:[\n ]*("leather"|"chainmail"|"plate")[\n ]*,[\n ]*"strength"[\n ]*:[\n ]*(0|[1-9][0-9]*)[\n ]*\}""", 124),
    (r"""\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"work"[\n ]*:[\n ]*\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"composer"[\n ]*:[\n ]*\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"functions"[\n ]*:[\n ]*\[("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*")(,("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"))*\][\n ]*\}[\n ]*\}[\n ]*,[\n ]*"recording_artists"[\n ]*:[\n ]*\[(\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"functions"[\n ]*:[\n ]*\[("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*")(,("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"))*\][\n ]*\})(,(\{[\n ]*"id"[\n ]*:[\n ]*(-)?((0|[1-9][0-9]*))(\.[0-9]+)?([eE][+-][0-9]+)?[\n ]*,[\n ]*"name"[\n ]*:[\n ]*"(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"[\n ]*,[\n ]*"functions"[\n ]*:[\n ]*\[("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*")(,("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"))*\][\n ]*\}))*\][\n ]*\}""", 196),
    (r"""choice 1|choice 2|car|truck|dog""", 15),
    (r"""long choice 1 giberrish blah blah blah|long choice 2 giberrish blah blah blah""", 38)
]

sync_client = OpenAI(
    base_url="http://localhost:8000/v1",
        api_key="token-abc123",
)
sync_client.chat.completions.create(
            model="unsloth/llama-3-8b-Instruct",
            messages=[
                {"role": "user", "content": "make me a character in json. make sure to format the JSON well with newlines."}
            ],
            extra_body={
                "guided_regex": prime_pattern,
                "guided_decoding_backend": 'outlines'
            }
        )
sync_client.chat.completions.create(
            model="unsloth/llama-3-8b-Instruct",
            messages=[
                {"role": "user", "content": "make me a character in json. make sure to format the JSON well with newlines."}
            ],
            extra_body={
                "guided_regex": prime_pattern,
                "guided_decoding_backend": 'faster-outlines'
            }
        )

async def make_request(client, backend, regex_pattern):
    regex, state_length = regex_pattern
    start_time = time.perf_counter()
    first_token_time = None
    end_time = None
    completion_tokens = 0
    
    try:
        stream = await client.chat.completions.create(
            model="unsloth/llama-3-8b-Instruct",
            messages=[
                {"role": "user", "content": "make me a character in json. make sure to format the JSON well with newlines."}
            ],
            extra_body={
                "guided_regex": regex,
                "guided_decoding_backend": backend
            },
            stream=True,
            max_tokens=150
        )
        
        content_chunks = []
        async for chunk in stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            if chunk.choices and chunk.choices[0].delta.content:
                content_chunks.append(chunk.choices[0].delta.content)
        
        end_time = time.perf_counter()
        
        full_response = ''.join(content_chunks)
        completion_tokens = len(tokenizer.encode(full_response))
        
        # Print completion for verification
        print(f"\nCompletion for {backend} (state length {state_length}):")
        print("=" * 80)
        print(full_response)
        print("=" * 80)
        
        # Use streaming time for token speed calculation
        streaming_time = end_time - first_token_time
        
        # If we didn't get completion_tokens from the API, estimate from content length
        if completion_tokens == 0:
            # Rough estimate: average English word is ~4 characters per token
            completion_tokens = len(full_response) // 4
            print(f"Estimated tokens (no usage stats): {completion_tokens}")
        else:
            print(f"Actual completion tokens: {completion_tokens}")
        
        tokens_per_second = completion_tokens / streaming_time if streaming_time > 0 else 0
        ttft = first_token_time - start_time
        total_time = end_time - start_time
        
        # Print timing information
        print(f"TTFT: {ttft:.3f}s")
        print(f"Streaming time: {streaming_time:.3f}s")
        print(f"Total time: {total_time:.3f}s")
        print(f"Tokens per second: {tokens_per_second:.1f}")
        print()
        
        return {
            'success': True,
            'ttft': ttft,
            'total_time': total_time,
            'streaming_time': streaming_time,
            'completion_tokens': completion_tokens,
            'tokens_per_second': tokens_per_second,
            'backend': backend,
            'state_length': state_length,
            'response_length': len(full_response),
            'response': full_response  # Include response for verification
        }
    except Exception as e:
        end_time = time.perf_counter()
        print(f"\nError in {backend} (state length {state_length}):")
        print(f"Error: {str(e)}")
        return {
            'success': False,
            'ttft': None,
            'total_time': end_time - start_time,
            'streaming_time': 0,
            'completion_tokens': 0,
            'tokens_per_second': 0,
            'backend': backend,
            'state_length': state_length,
            'error': str(e),
            'response_length': 0,
            'response': None
        }

async def benchmark_backend(client, backend, regex_pattern, num_requests=10):
    tasks = [make_request(client, backend, regex_pattern) for _ in range(num_requests)]
    return await asyncio.gather(*tasks)

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    
    backends = ['outlines', 'faster-outlines']
    num_requests = 1  # Reduced number since we're testing multiple patterns
    all_results = []
    
    for backend in backends:
        for regex_pattern in REGEX_PATTERNS:
            print(f"\nBenchmarking {backend} with state length {regex_pattern[1]}...")
            print(f"Pattern: {regex_pattern[0][:100]}..." if len(regex_pattern[0]) > 100 else regex_pattern[0])
            
            results = await benchmark_backend(client, backend, regex_pattern, num_requests)
            
            # Calculate statistics for this combination
            successful_results = [r for r in results if r['success']]
            
            if successful_results:
                ttft_times = [r['ttft'] for r in successful_results]
                total_times = [r['total_time'] for r in successful_results]
                streaming_times = [r['streaming_time'] for r in successful_results]
                token_counts = [r['completion_tokens'] for r in successful_results]
                tokens_per_second = [r['tokens_per_second'] for r in successful_results]
                response_lengths = [r['response_length'] for r in successful_results]
                
                stats = {
                    'backend': backend,
                    'state_length': regex_pattern[1],
                    'pattern': regex_pattern[0],
                    'total_requests': len(results),
                    'successful_requests': len(successful_results),
                    'failed_requests': len(results) - len(successful_results),
                    'mean_ttft': statistics.mean(ttft_times),
                    'median_ttft': statistics.median(ttft_times),
                    'min_ttft': min(ttft_times),
                    'max_ttft': max(ttft_times),
                    'mean_total_time': statistics.mean(total_times),
                    'median_total_time': statistics.median(total_times),
                    'min_total_time': min(total_times),
                    'max_total_time': max(total_times),
                    'mean_streaming_time': statistics.mean(streaming_times),
                    'mean_completion_tokens': statistics.mean(token_counts),
                    'mean_response_length': statistics.mean(response_lengths),
                    'mean_tokens_per_second': statistics.mean(tokens_per_second),
                    'median_tokens_per_second': statistics.median(tokens_per_second),
                    'min_tokens_per_second': min(tokens_per_second),
                    'max_tokens_per_second': max(tokens_per_second),
                    'example_response': successful_results[0]['response'],  # Store an example response
                    'raw_ttft': ttft_times,
                    'raw_total_times': total_times,
                    'raw_streaming_times': streaming_times,
                    'raw_tokens_per_second': tokens_per_second,
                    'timestamp': datetime.now().isoformat()
                }
                all_results.append(stats)
                
                print(f"\nPattern Summary:")
                print(f"Success Rate: {len(successful_results)}/{len(results)}")
                print(f"Mean TTFT: {stats['mean_ttft']:.2f}s")
                print(f"Mean Total Time: {stats['mean_total_time']:.2f}s")
                print(f"Mean Streaming Time: {stats['mean_streaming_time']:.2f}s")
                print(f"Mean Response Length: {stats['mean_response_length']:.1f} chars")
                print(f"Mean Tokens: {stats['mean_completion_tokens']:.1f}")
                print(f"Mean Tokens/s: {stats['mean_tokens_per_second']:.2f}")
                print("-" * 80)
    
    # Save results to JSON file
    with open('regex_benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print("\nOverall Summary Table:")
    print("=" * 120)
    print(f"{'Backend':<15} {'Mean TTFT':<12} {'Mean Stream':<12} {'Total':<12} {'Chars':<8} {'Tokens':<8} {'Tokens/s':<12} {'Success'}")
    print("-" * 120)
    
    for backend in backends:
        backend_results = [r for r in all_results if r['backend'] == backend]
        if backend_results:
            avg_ttft = statistics.mean(r['mean_ttft'] for r in backend_results)
            avg_stream = statistics.mean(r['mean_streaming_time'] for r in backend_results)
            avg_total = statistics.mean(r['mean_total_time'] for r in backend_results)
            avg_chars = statistics.mean(r['mean_response_length'] for r in backend_results)
            avg_tokens = statistics.mean(r['mean_completion_tokens'] for r in backend_results)
            avg_tps = statistics.mean(r['mean_tokens_per_second'] for r in backend_results)
            success_rate = sum(r['successful_requests'] for r in backend_results) / sum(r['total_requests'] for r in backend_results) * 100
            
            print(f"{backend:<15} {avg_ttft:<12.3f} {avg_stream:<12.3f} {avg_total:<12.3f} {avg_chars:<8.1f} {avg_tokens:<8.1f} {avg_tps:<12.1f} {success_rate:>6.1f}%")
    
    print("\nResults saved to regex_benchmark_results.json")

if __name__ == "__main__":
    asyncio.run(main())