from .regex import create_fsm_index_tokenizer
import interegular

class RegexGuide():
    """Guide to generate text in the language of a regular expression."""

    initial_state = 0

    def __init__(self, regex_string: str, tokenizer: "Tokenizer"):
        (
            self.fsm,
            self.empty_token_ids,
            fsm_finals,
        ) = create_fsm_index_tokenizer(regex_string, tokenizer)
        self.eos_token_id = tokenizer.eos_token_id
        self.final_states = fsm_finals | {-1}
        
    @property
    def states_to_token_maps(self):
        self.fsm.get_states_to_token_subsets()

    def get_next_instruction(self, state: int):
        """Return the next instruction for guided generation.

        The initialization of the guide builds an index which maps FSM states to a
        map from authorized tokens to the state in which the guide needs to move
        if said token is generated. Therefore the authorized tokens at the
        current state are the keys of the map returned by the value of the index
        for current state.

        If the current state is not contained in the end this means that we are
        in a final state of the guide. We only authorize EOS tokens in the final
        state.

        Parameters
        ----------
        state
            The current state of the guide.

        Returns
        -------
        A `Generate` instance that contains the model and the allowed token ids.

        """
        return self.fsm.get_next_instruction(state)

    def get_next_state(self, state: int, token_id: int) -> int:
        """Update the state of the guide.

        We use the index to determine to which state the guide should transition
        given the token that was just generated.

        Parameters
        ----------
        state
            The current state of the guide.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the guide.

        """
        return self.fsm.get_next_state(state, token_id)

    @classmethod
    def from_interegular_fsm(
        cls, interegular_fsm: interegular.fsm.FSM, tokenizer: "Tokenizer"
    ):
        from_interegular_instance = cls.__new__(cls)

        byte_fsm = make_byte_level_fsm(interegular_fsm.reduce(), keep_utf8=True)
        regex_fsm, _ = make_deterministic_fsm(byte_fsm)

        (
            from_interegular_instance.fsm,
            from_interegular_instance.empty_token_ids,
            fsm_finals,
        ) = create_fsm_index_tokenizer(regex_fsm, tokenizer)

        from_interegular_instance.eos_token_id = tokenizer.eos_token_id
        from_interegular_instance.final_states = fsm_finals | {-1}
        return from_interegular_instance
