from functools import lru_cache


@lru_cache
def reduced_vocabulary(tokenizer):
    """Create a map from decoded vocabulary tokens to lists of equivalent token ids."""
    vocabulary = {}
    empty_token_ids = set()
    for token, token_idx in tokenizer.vocabulary.items():
        if token in tokenizer.special_tokens:
            continue

        token_str = tokenizer.convert_token_to_string(token)

        if token_str:
            # Ensure the key exists with an empty list, then append
            vocabulary.setdefault(token_str, []).append(token_idx)
        else:
            empty_token_ids.add(token_idx)

    return vocabulary, empty_token_ids
