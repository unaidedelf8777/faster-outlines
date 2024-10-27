from functools import lru_cache

SPIECE_UNDERLINE = "\u2581"

def convert_token_to_string(token: str) -> str:

    if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
        return " " + token

    return token

@lru_cache
def reduced_vocabulary(tokenizer, _convert_token_to_string=convert_token_to_string):
    """Create a map from decoded vocabulary tokens to lists of equivalent token ids."""
    vocabulary = {}
    empty_token_ids = set()
    special_tokens = set(tokenizer.all_special_tokens)
    for token, token_idx in tokenizer.get_vocab().items():
        if token in special_tokens:
            continue

        token_str = convert_token_to_string(token)

        if token_str:
            # Ensure the key exists with an empty list, then append
            vocabulary.setdefault(token_str, []).append(token_idx)
        else:
            empty_token_ids.add(token_idx)

    return vocabulary, empty_token_ids