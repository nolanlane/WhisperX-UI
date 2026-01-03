import sys
import os
import warnings

print("[UMS] Loading sitecustomize.py - Applying global patches...")

# 1. Suppress annoying deprecation warnings from pyannote/torchaudio
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.set_audio_backend.*")

# 2. FIX: Global monkey patch for BasicSR compatibility with torchvision >= 0.16
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
    os.environ["BASICSR_PATCHED"] = "1"
except (ImportError, Exception):
    pass

# 3. FIX: NumPy 2.0 compatibility patch for legacy libraries
try:
    import numpy as np
    # Many older libraries (BasicSR, GFPGAN) use these removed aliases
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'bool'):
        np.bool = bool
    if not hasattr(np, 'complex'):
        np.complex = complex
    if not hasattr(np, 'object'):
        np.object = object
except ImportError:
    pass

# 4. FIX: Patch gradio_client for compatibility with schemas using additionalProperties: True
# This fixes crashes in gradio 4.44.1 when schemas contain boolean values (True/False)
try:
    # Handle different import locations/versions safely
    import gradio_client.utils as client_utils

    # Patch 1: get_type (used in some contexts)
    if hasattr(client_utils, 'get_type'):
        _orig_get_type = client_utils.get_type

        def _patched_get_type(schema):
            if isinstance(schema, bool):
                return "Any"
            return _orig_get_type(schema)

        client_utils.get_type = _patched_get_type

    # Patch 2: _json_schema_to_python_type (used in others)
    if hasattr(client_utils, '_json_schema_to_python_type'):
        _orig_json_schema_to_python_type = client_utils._json_schema_to_python_type

        def _patched_json_schema_to_python_type(schema, defs):
            if schema is True:
                return "Any"
            if schema is False:
                return "None"
            return _orig_json_schema_to_python_type(schema, defs)

        client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type

except ImportError:
    pass
except Exception as e:
    print(f"[UMS] Warning: Failed to patch gradio_client: {e}")

# 4. FIX: Mitigate CVE-2025-32434 (RCE in torch.load)
# Globally force weights_only=True for torch.load if it's not specified
import torch
_orig_torch_load = torch.load
def _patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    if weights_only is None:
        # In Jan 2026, we enforce this for all untrusted/legacy loads
        weights_only = True
    return _orig_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=weights_only, **kwargs)
torch.load = _patched_torch_load

print("[UMS] sitecustomize.py loaded successfully.")
