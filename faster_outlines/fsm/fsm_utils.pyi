from typing import Any, Dict, List, Optional, Set

class Write:
    """Write instruction for direct token sequences.

    Attributes:
        tokens (List[int]): The sequence of tokens to be added to the current
            sequence by the generation process.
    """
    tokens: List[int]

class Generate:
    """Generate instruction for constrained token choices.

    Attributes:
        tokens (Optional[List[int]]): The tokens that lead to a valid completion
            if generated. A value of None indicates that all tokens are allowed.
    """
    tokens: Optional[List[int]]

class TokenVocabulary:
    """TokenVocabulary manages a vocabulary of tokens, with serialization support.

    Attributes:
        eos_token_id (int): The end-of-sequence token identifier.
    """

    def __init__(
        self,
        vocab_dict: Dict[str, List[int]],
        eos_token_id: int,
        special_tokens: Set[str]
    ) -> None:
        """Initialize the TokenVocabulary.

        Args:
            vocab_dict (Dict[str, List[int]]): Dictionary mapping token strings
                to their corresponding list of integer values.
            eos_token_id (int): The end-of-sequence token identifier.
            special_tokens (Set[str]): Set of tokens to exclude from processing.
        """
        ...

    @property
    def eos_token_id(self) -> int:
        """Get the end-of-sequence token identifier.

        Returns:
            int: The EOS token ID.
        """
        ...

    def add_token(self, token: str, values: List[int]) -> None:
        """Add a token to the vocabulary.

        Args:
            token (str): The token string to add.
            values (List[int]): Integer values associated with the token.
        """
        ...

    def remove_token(self, token: str) -> Optional[List[int]]:
        """Remove a token from the vocabulary.

        Args:
            token (str): The token string to remove.

        Returns:
            Optional[List[int]]: Values of removed token, or None if not found.
        """
        ...

    def __len__(self) -> int:
        """Get number of tokens in vocabulary.

        Returns:
            int: Number of tokens.
        """
        ...

    def is_empty(self) -> bool:
        """Check if vocabulary is empty.

        Returns:
            bool: True if empty, False otherwise.
        """
        ...

def create_fsm_index_end_to_end_rs(
    fsm_info: Dict[str, Any],
    vocabulary: TokenVocabulary,
) -> 'LazyFSMIndex':
    """Create a LazyFSMIndex instance.

    Args:
        fsm_info: FSM definition and configuration.
        vocabulary: Token vocabulary for the FSM.
    
    Returns:
        LazyFSMIndex: New FSM index instance.
    """
    ...

def get_cached_fsm(hash: int) -> Dict[str, Any]:
    """Retrieve cached FSM by hash.

    Args:
        hash (int): Cache key hash.

    Returns:
        Dict[str, Any]: Cached FSM data if found.

    Raises:
        ValueError: If FSM not found in cache.
    """
    ...

def get_fsm_cache_key(pattern: str, vocabulary: TokenVocabulary) -> int:
    """Generate cache key for pattern and vocabulary combination.

    Args:
        pattern (str): FSM pattern string.
        vocabulary (TokenVocabulary): Token vocabulary.

    Returns:
        int: Cache key hash.
    """
    ...

class LazyFSMIndex:
    """Lazily computed FSM index for efficient pattern matching.
    
    Provides asynchronous computation of state transitions with caching.
    """

    def get_next_state(self, state: int, token_id: int) -> Optional[int]:
        """Get next state for given current state and token.

        Args:
            state: Current state ID.
            token_id: Input token ID.

        Returns:
            Optional[int]: Next state ID if valid transition exists.
        """
        ...

    def get_next_instruction(self, state: int) -> 'Write | Generate':
        """Get next instruction for pattern-guided generation.

        Args:
            state: Current state ID.

        Returns:
            Union[Write, Generate]: Next instruction for generation.
        """
        ...

    def collect_finished_states(self) -> Dict[int, Dict[int, int]]:
        """Collect newly computed state transitions.

        Returns:
            Dict[int, Dict[int, int]]: Map of state ID to transitions.

        Raises:
            ValueError: If collection fails.
        """
        ...

    def await_state(self, state_index: int) -> None:
        """Wait for specific state computation to complete.

        Args:
            state_index: State ID to wait for.

        Raises:
            ValueError: If state index invalid.
        """
        ...

    def await_finished(self) -> None:
        """Wait for all state computations to complete."""
        ...

    def get_allowed_token_ids(self, state: int) -> List[int]:
        """Get allowed tokens for state (debug utility).

        Args:
            state: State ID to check.

        Returns:
            List[int]: Allowed token IDs.
        """
        ...

    def __repr__(self) -> str:
        """Get string representation.

        Returns:
            str: Debug representation of FSM index.
        """
        ...