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
