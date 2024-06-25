import time
import faster_outlines
from faster_outlines.fsm import FsmTokenizer
from transformers import AutoTokenizer
from json import dumps as json_dumps

s = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            "required": ["location", "format"],
        },
    },
    {
        "name": "get_reservation",
        "description": "Retrieve a reservation at a restaurant",
        "parameters": {
            "type": "object",
            "properties": {
                "restaurant_name": {
                    "type": "string",
                    "description": "The name of the restaurant for which the reservation is made",
                },
                "reservation_date": {
                    "type": "string",
                    "format": "date",
                    "description": "The date of the reservation in YYYY-MM-DD format",
                },
                "reservation_time": {
                    "type": "string",
                    "format": "time",
                    "description": "The time of the reservation in HH:MM format",
                },
                "party_size": {
                    "type": "integer",
                    "description": "The number of people included in the reservation",
                },
                "contact_number": {
                    "type": "integer",
                    "description": "The contact phone number for the reservation confirmation",
                },
            },
            "required": ["restaurant_name", "reservation_time"],
        },
    },
]

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


# A function to perform the benchmark test
def test_benchmark_compile_fsm():
    """Benchmark the numba compilation time without mocker."""

    # Reload the module to apply the patched njit
    from faster_outlines.fsm.regex import create_fsm_index_tokenizer

    tokenizer = FsmTokenizer(
        AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
    )

    # Benchmark phase
    for pattern in test_patterns:
        total_time = 0
        iterations = 6  # Set a constant number of iterations
        for i in range(1, iterations + 1):
            print("starting timer")
            start_time = time.perf_counter()
            fsm, empty_token_ids = create_fsm_index_tokenizer(pattern, tokenizer)
            fsm.get_states_to_token_subsets()
            end_time = time.perf_counter()
            computation_time = end_time - start_time
            total_time += computation_time
            
            
            

            print(f"Time taken for Rust: {computation_time} seconds")
            print("====================================")
            print(f"first state: {0}")
            # Uncomment the following line if the decoding function and the `allowed_token_ids` method are correctly defined and relevant
            print("====================================")
            time.sleep(0.5)

        average_time = total_time / iterations
        print("************************************")
        print(f"Average time for pattern '{pattern}': {average_time} seconds")
        print("************************************")
        


# Run the benchmark test
test_benchmark_compile_fsm()