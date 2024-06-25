from pydantic import BaseModel, ValidationError
from typing import List
from transformers import PreTrainedTokenizer
import json
from .utils import (
    calc_fn_tokens,
    find_variant_tokens,
    find_tokens_with_char,
    get_int_tokens,
    get_float_tokens,
)
# Ensure the necessary token finding functions are imported or defined here


class TokenMap(BaseModel):
    """
    Default token mappings for Mistral, Mixtral, and derivatives.
    """

    open_bracket: List[int] = []
    close_bracket: List[int] = []
    comma: List[int] = []
    quote: List[int] = []
    quote_comma: List[int] = []
    eov: List[int] = []
    quote_banned: List[int] = []
    list_open: List[int] = []
    list_close: List[int] = []
    integer_tokens: List[int] = []
    float_tokens: List[int] = []
    colon: List[int] = []
    space: List[int] = []
    name: List[int] = []

    @classmethod
    def build(cls, tokenizer: PreTrainedTokenizer):
        instance = cls(
            open_bracket=find_variant_tokens(tokenizer, "{"),
            close_bracket=calc_fn_tokens("}", "a", tokenizer),
            comma=find_variant_tokens(tokenizer, ","),
            quote=calc_fn_tokens('"', "a", tokenizer),
            quote_comma=calc_fn_tokens('",', "a", tokenizer),
            eov=calc_fn_tokens('",', "a", tokenizer)
            + calc_fn_tokens(",", "a", tokenizer),
            quote_banned=find_tokens_with_char(tokenizer, ["'", '"']),
            list_open=find_variant_tokens(tokenizer, "["),
            list_close=find_variant_tokens(tokenizer, "]"),
            integer_tokens=get_int_tokens(tokenizer),
            float_tokens=get_float_tokens(tokenizer),
            colon=find_variant_tokens(tokenizer, ":"),
            space=tokenizer.encode(" ", add_special_tokens=False)
            + tokenizer.encode("  ", add_special_tokens=False)
            + [tokenizer.encode("> ", add_special_tokens=False)[1]]
            if len(tokenizer.encode("> ", add_special_tokens=False)) == 2
            else [],
            name=tokenizer.encode("name", add_special_tokens=False),
        )
        return instance

    @classmethod
    def from_json_file(cls, file_path: str):
        """
        Load token map data from a JSON file and return an instance of TokenMap.

        Args:
            file_path (str): The path to the JSON file containing the token map data.

        Returns:
            TokenMap: An instance of TokenMap initialized with data from the JSON file.
        """
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            return cls(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {file_path}") from e
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
        except ValidationError as e:
            raise ValueError(f"Error validating TokenMap data from {file_path}: {e}")


class MistralTokenMap(TokenMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
