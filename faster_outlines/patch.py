from .fsm import (
    create_fsm_index_tokenizer,
    Write,
    Generate
)


def get_next_instruction_regexfsm(
    self,
    state: int
    ):
    # self.states_to_token_maps = a LazyFSMIndex instance
    return self.states_to_token_maps.get_next_instruction(state)

def get_next_state_regexfsm(
    self,
    state: int,
    token_id: int
    ):
    # self.states_to_token_maps = a LazyFSMIndex instance
    return self.states_to_token_maps.get_next_state(state, token_id)
    
def patch(outlines_module): 
    """
    Patch the vanilla `outlines` module such that it uses the `faster-outlines` backend.

    This function replaces the original create_states_mapping function in 
    outlines_module.fsm.guide with the create_fsm_index_tokenizer function. 
    This modification is done in-place and affects the module globally.
    If needed, this function also returns the modified module, in case
    your design pattern prefers this.

    Parameters:
    -----------
     - outlines_module : module
        The outlines module to be patched. This should be the module object,
        not a string name.

    Returns:
    --------
     - module
        The patched outlines module. Note that this is the same object that
        was passed in, modified in-place.

    Usage:
    --------------
    >>> import outlines
    >>> patched_outlines = patch(outlines) 
    >>> # Or just `patch(outlines)`
    >>> # Now, all uses of the module will use the backend from `faster_outlines` instead of default.
    """
    try:
        outlines_module.fsm.guide.Write = Write
        outlines_module.fsm.guide.Generate = Generate
        outlines_module.fsm.guide.create_states_mapping = create_fsm_index_tokenizer
        outlines_module.fsm.guide.RegexGuide.get_next_state = get_next_state_regexfsm
        outlines_module.fsm.guide.RegexGuide.get_next_instruction = get_next_instruction_regexfsm
        
    except Exception as e:
        print("ERROR! patching outlines module failed.")
        raise e
    
    return outlines_module
