import unicodedata


def find_variant_tokens(tokenizer, char):
    tokens_containing_char = set()
    normalized_char = unicodedata.normalize("NFKC", char)

    for i in range(4):
        if i == 0:
            t = tokenizer.encode(char, add_special_tokens=False)
            if len(t) == 1:
                tokens_containing_char.update(t)

        if i == 1:
            t = tokenizer.encode(" " + char, add_special_tokens=False)
            if len(t) == 1:
                tokens_containing_char.update(t)

        if i == 2:
            t = tokenizer.encode(char + " ", add_special_tokens=False)
            if len(t) == 1:
                tokens_containing_char.update(t)

        if i == 3:
            t = tokenizer.encode(" " + char + " ", add_special_tokens=False)
            if len(t) == 1:
                tokens_containing_char.update(t)

        if i == 4:
            # sometimes, a char is tokenized as a different token when it is against another token.
            # this way we get that one to.
            t = tokenizer.encode(char, add_special_tokens=False)
            if len(t) == 1:
                t = tokenizer.encode("a" + char, add_special_tokens=False)
                if len(t) == 2:
                    tokens_containing_char.update([t[1]])

    # Iterate through the tokenizer's vocabulary
    for token, index in tokenizer.get_vocab().items():
        # Normalize the token using NFKC normalization
        normalized_token = unicodedata.normalize("NFKC", token)
        # Check if the normalized character is in the normalized token
        if normalized_char == normalized_token.strip():
            tokens_containing_char.update([index])

    return list(tokens_containing_char)


def find_tokens_with_char(tokenizer, chars):
    tokens_containing_chars = set()

    # Normalize and iterate over each character provided
    for char in chars:
        normalized_char = unicodedata.normalize("NFKC", char)

        # Iterate through the tokenizer's vocabulary
        for token, index in tokenizer.get_vocab().items():
            # Normalize the token using NFKC normalization
            normalized_token = unicodedata.normalize("NFKC", token)
            # Check if the normalized character is part of the normalized token
            if normalized_char in normalized_token:
                tokens_containing_chars.add(index)

    return list(tokens_containing_chars)


def get_int_tokens(tokenizer):
    allowed_token_ids = []
    for token_str, token_id in tokenizer.get_vocab().items():
        token_str = token_str.strip()

        if token_str == "" or (
            all(c.isdigit() for c in token_str) and token_str.count(".") == 0
        ):
            allowed_token_ids.append(token_id)
    return allowed_token_ids


def get_float_tokens(tokenizer):
    allowed_token_ids = []
    for token_str, token_id in tokenizer.get_vocab().items():
        token_str = token_str.strip()

        if token_str == "" or (
            all(c.isdigit() or c == "." for c in token_str)
            and token_str.count(".") <= 1
        ):
            allowed_token_ids.append(token_id)
    return allowed_token_ids


def calc_fn_tokens(token, ctx_token, tokenizer, nested=False):
    ctx_tok = tokenizer.encode(ctx_token, add_special_tokens=False)

    with_ctx = tokenizer.encode(ctx_token + token, add_special_tokens=False)

    without_ctx = tokenizer.encode(token, add_special_tokens=False)

    final = set()

    # are they the same length?
    if len(with_ctx) == len(without_ctx):
        final.append(without_ctx)
    else:
        if len(ctx_tok) == 1:
            with_ctx.remove(ctx_tok[0])
            if nested:
                final.update([with_ctx])
                final.update([without_ctx])
            else:
                final.update(with_ctx)

    return list(final)
