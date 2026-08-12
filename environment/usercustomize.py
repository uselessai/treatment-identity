"""Compatibility layer present in the retained RRTN environment."""

# SciPy >= 1.8 removed scipy.finfo from the top-level namespace.  The audited
# KAIR BlindSR subject still calls scipy.finfo(float), so preserve the alias
# that was present when the portability campaign ran.
try:
    import numpy as _np
    import scipy as _scipy

    if not hasattr(_scipy, "finfo"):
        _scipy.finfo = _np.finfo
except Exception:
    pass
