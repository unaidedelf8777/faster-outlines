import sys
from .fsm import RegexGuide, Write, Generate


def patch(outlines_module, save_to_sys_modules=True):
    """
    Patch the vanilla `outlines` module to use the `faster-outlines` backend.

    This function modifies the `outlines_module` in-place by replacing specific
    functions and classes with the patched versions. Optionally, it can save
    the modified module back into `sys.modules`.

    Parameters:
    -----------
     - outlines_module : module
        The outlines module to be patched. This should be the actual module object,
        not a string name.

     - save_to_sys_modules : bool, optional
        If True (default), the modified outlines module will be saved to sys.modules.
        If False, the module will be patched but not saved to sys.modules.

    Returns:
    --------
     - module
        The patched outlines module. This is the same object that was passed in,
        modified in-place.

    Raises:
    -------
     - ImportError
        If the outlines module is not found in sys.modules.

    Usage:
    --------------
    >>> import outlines
    >>> patched_outlines = patch(outlines)
    >>> # Now, all uses of the module will use the backend from `faster_outlines`.
    """

    try:
        if "outlines" not in sys.modules:
            raise ImportError(
                "The outlines module is not loaded in sys.modules. Please import it before patching."
            )

        outlines_module.fsm.guide.Write = Write
        outlines_module.fsm.guide.Generate = Generate

        outlines_module.fsm.guide.RegexGuide = RegexGuide

        if save_to_sys_modules:
            sys.modules["outlines"] = outlines_module
            sys.modules["outlines.fsm"] = outlines_module.fsm
            sys.modules["outlines.fsm.guide"] = outlines_module.fsm.guide

    except Exception as e:
        print("ERROR! Patching outlines module failed.")
        raise e

    return outlines_module
