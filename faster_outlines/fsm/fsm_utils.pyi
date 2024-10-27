from typing import Any, Dict, List, Optional, Any

class Write:
    """Write instruction.

    Attributes
    ----------
    tokens
        The sequence of tokens to be added to the current sequence by the
        generation process.

    """

    tokens: List[int]

class Generate:
    """Generate instruction

    Attributes
    ----------
    tokens
        The tokens that lead to a valid completion if generated.  A value
        of ``None`` indicates that all tokens are allowed.
    """

    tokens: Optional[List[int]]

class TokenVocabulary:
    """
    TokenVocabulary is a class that manages a vocabulary of tokens, allowing
    addition and removal of tokens, and serialization for persistence.

    Attributes:
        eos_token_id (int): The end-of-sequence token identifier.
    """

    def __init__(
        self,
        vocab_dict: Optional[Dict[str, List[int]]] = ...,
        eos_token_id: Optional[int] = ...
    ) -> None:
        """
        Initializes the TokenVocabulary.

        Args:
            vocab_dict (Optional[Dict[str, List[int]]]): A dictionary mapping
                token strings to their corresponding list of integer values.
                If `None`, an empty vocabulary is initialized for deserialization.
            eos_token_id (Optional[int]): The end-of-sequence token identifier.
                Must be provided if `vocab_dict` is provided.

        Raises:
            ValueError: If only one of `vocab_dict` or `eos_token_id` is provided.
        """
        ...

    @property
    def eos_token_id(self) -> int:
        """
        Gets the end-of-sequence token identifier.

        Returns:
            int: The EOS token ID.
        """
        ...

    def add_token(self, token: str, values: List[int]) -> None:
        """
        Adds a new token to the vocabulary.

        Args:
            token (str): The token string to add.
            values (List[int]): A list of integer values associated with the token.
        """
        ...

    def remove_token(self, token: str) -> Optional[List[int]]:
        """
        Removes a token from the vocabulary.

        Args:
            token (str): The token string to remove.

        Returns:
            Optional[List[int]]: The list of integer values associated with the removed token,
            or `None` if the token was not found.
        """
        ...

    def __len__(self) -> int:
        """
        Returns the number of tokens in the vocabulary.

        Returns:
            int: The number of tokens.
        """
        ...

    def is_empty(self) -> bool:
        """
        Checks whether the vocabulary is empty.

        Returns:
            bool: `True` if the vocabulary is empty, `False` otherwise.
        """
        ...

    def __getstate__(self) -> bytes:
        """
        Serializes the TokenVocabulary for pickling.

        Returns:
            bytes: The serialized state of the vocabulary.

        Raises:
            ValueError: If serialization fails.
        """
        ...

    def __setstate__(self, state: bytes) -> None:
        """
        Deserializes the TokenVocabulary from a pickled state.

        Args:
            state (bytes): The serialized state to deserialize.

        Raises:
            ValueError: If deserialization fails.
        """
        ...

def create_fsm_index_end_to_end_rs(
    fsm_info: Dict[str, Any],
    vocabulary: TokenVocabulary,
    eos_token_id: int
) -> 'LazyFSMIndex':
    """
    create_fsm_index_end_to_end_rs(fsm_info, vocabulary, eos_token_id) -> LazyFSMIndex

    Creates a LazyFSMIndex instance using the provided FSMInfo, TokenVocabulary, and eos_token_id.
    """
    ...

def start_cache_receiver() -> None:
    """
    Starts the cache receiver thread.
    """
    ...

def stop_cache_receiver() -> None:
    """
    Stops the cache receiver thread.
    """
    ...

class LazyFSMIndex:
    """
    A class representing a lazily computed FSM index.

    Methods:
        get_states_to_token_subsets() -> Dict[int, Dict[int, int]]
        collect_finished_states() -> Dict[int, Dict[int, int]]
        is_final_state(state: int) -> bool
        is_computing_finished() -> bool
        await_state(state_index: int) -> None
        await_finished() -> None
        allowed_token_ids(state: int) -> List[int]
        __repr__() -> str
    """

    def get_states_to_token_subsets(self) -> Dict[int, Dict[int, int]]:
        """
        Returns the mapping of states to token subsets.

        Returns:
            Dict[int, Dict[int, int]]: The states to token subsets mapping.
        """
        ...

    def collect_finished_states(self) -> Dict[int, Dict[int, int]]:
        """
        Collects and returns the finished states.

        This is part of a more advanced, asynchronous interface to the object.
        Note the function is not async, rather it returns instantly with the 
        data from finished states, not the full index.

        Returns:
            Dict[int, Dict[int, int]]: The finished states mapping.
        """
        ...

    def is_final_state(self, state: int) -> bool:
        """
        Checks if the given state is a final state.

        Args:
            state (int): The state to check.

        Returns:
            bool: True if final state, False otherwise.
        """
        ...

    def is_computing_finished(self) -> bool:
        """
        Checks if the computation is finished.

        Returns:
            bool: True if finished, False otherwise.
        """
        ...

    def await_state(self, state_index: int) -> None:
        """
        Waits until the specified state is computed.

        Args:
            state_index (int): The index of the state to await.
        """
        ...

    def await_finished(self) -> None:
        """
        Waits until the entire computation is finished.
        """
        ...

    def allowed_token_ids(self, state: int) -> List[int]:
        """
        Returns the allowed token IDs for the given state.

        Args:
            state (int): The state for which to get allowed token IDs.

        Returns:
            List[int]: A list of allowed token IDs.
        """
        ...
    
    def get_next_instruction(self, state: int) -> "Instruction":
        """
        
        Returns the next Instruction in the FSM.
        
        Args:
            state (int): The state for which to get the instruction.

        Instructions have two types: 
            - Write:
                *tokens* Optional[List[int]]: The tokens to write to the Language model context window.
                This can include EOS token id if the fsm reaches the final state.
            
            - Generate:
                *tokens* List[int]: The allowed token ID's which are each valid to be chosen 
                at this state. the language model should choose one.
        """
        ...
    
    def get_next_state(self, state: int, token_id: int) -> int:
        """
        Args:
            state (int), token_id (int)
        
        returns:
            int, the next state of the fsm, given the state and transition (token_id)
        """
        ...
        
    def __repr__(self) -> str:
        """
        Returns a string representation of the LazyFSMIndex instance.

        Returns:
            str: The string representation.
        """
        ...
