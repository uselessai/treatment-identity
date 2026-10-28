"""The sample size a project configures, fitted to the fixture it is given.

An adapter asks the loader for the patch size the project's own configuration
sets. Some checks run on fixtures smaller than that patch; there the adapter
asks for the largest patch the fixture holds, because a loader asked to crop
more than the image has either raises or, in one subject, silently substitutes
a tile of random colour. The checks that concern size and content run on
fixtures larger than the configured patch, so they see the configured value.
"""
from __future__ import annotations

from pathlib import Path

import cv2


def shorter_side(root) -> int:
    """Shorter side of the first fixture frame under ``root``."""
    first = next(iter(sorted(Path(root).rglob("*.png"))))
    h, w = cv2.imread(str(first)).shape[:2]
    return min(h, w)


def fitted(configured: int, root, multiple: int = 1) -> int:
    """The configured size if the fixture holds it, else the largest that fits."""
    side = shorter_side(root)
    if side >= configured:
        return configured
    return max(multiple, side - side % multiple)
