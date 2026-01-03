import sys
import os
import warnings

print("[UMS] Loading sitecustomize.py - Applying global patches...")

# 1. Suppress annoying deprecation warnings from pyannote/torchaudio
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.set_audio_backend.*")

# 2. FIX: NumPy compatibility patch for legacy libraries
try:
    import numpy as np
    # Some older dependencies might still use these removed aliases in NumPy 2.0
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
except Exception as e:
    print(f"[UMS] Warning: Failed to patch NumPy: {e}")

print("[UMS] sitecustomize.py loaded successfully.")
