"""nki-kernel-lab: Trainium2 kernels written in NKI, verified on the CPU simulator."""
import os

# Target Trainium2 (NeuronCore-v3) with one logical core unless the caller overrides.
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "1")

import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning, module="nki")

__version__ = "0.1.0"
