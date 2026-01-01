import sys
import os

# FIX: Global monkey patch for BasicSR compatibility with torchvision >= 0.16
# This file is automatically imported by all Python processes in the environment
# if the directory is in PYTHONPATH.
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
    os.environ["BASICSR_PATCHED"] = "1"
except ImportError:
    pass
except Exception:
    pass

# FIX: Patch gradio_client for compatibility with schemas using additionalProperties: True
try:
    from gradio_client import utils

    _orig_get_type = utils.get_type

    def _patched_get_type(schema):
        if isinstance(schema, bool):
            # Fallback for boolean schema (e.g. additionalProperties: True)
            return "Any"
        return _orig_get_type(schema)

    utils.get_type = _patched_get_type
except ImportError:
    pass
