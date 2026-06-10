"""nemopy — A column-vector-first NumPy wrapper."""

__version__ = "0.1.0"

# Layer 1: Re-export NumPy's full public namespace
from numpy import *  # noqa: F403, F401

# Layer 1b: Re-export key NumPy submodules so np.linalg, np.random, etc. work
import numpy.linalg as linalg    # noqa: F401
import numpy.random as random    # noqa: F401
import numpy.fft as fft          # noqa: F401
import numpy.ma as ma            # noqa: F401

# Layer 2: Override with nemopy's own types and constructors
from nemopy._core import ColVec, Mat, ShapeError, ConventionWarning  # noqa: F401
from nemopy._constructors import _c, _m, mat, eye, as_col, as_mat  # noqa: F401
from nemopy._markov import markov, ctmc  # noqa: F401
from nemopy._ahp import (  # noqa: F401
    ahp_matrix,
    ahp_synthesize,
    ahp_aggregate,
    anp_supermatrix,
)
from nemopy import _operators  # noqa: F401
from nemopy import _stats  # noqa: F401
from nemopy import _decompositions  # noqa: F401
from nemopy import _elimination  # noqa: F401

__all__ = [
    "_c",
    "_m",
    "mat",
    "eye",
    "as_col",
    "as_mat",
    "markov",
    "ctmc",
    "ahp_matrix",
    "ahp_synthesize",
    "ahp_aggregate",
    "anp_supermatrix",
    "ColVec",
    "Mat",
    "ShapeError",
    "ConventionWarning",
]
