from __future__ import annotations

import numpy as np

from numpy.typing import NDArray
from typing import TypeAlias


# ==============================================================================
# Array typing aliases
# ==============================================================================

FloatArray: TypeAlias = NDArray[np.float64]

IntArray: TypeAlias = NDArray[np.int_]

BoolArray: TypeAlias = NDArray[np.bool_]