"""Put the random number generators an audited loader may draw from into a
known state, and record honestly which ones could not be reached.

This module exists because of a defect in this protocol, not in the code it
audits. An early version passed ``LoaderSpec.seed`` to the adapter and never
applied it. Loaders that select their temporal window deterministically were
unaffected, so the two failing pipelines reported the same result on every run.
The one loader that samples its window at random reported a different frame
coverage each time it was probed --- 14, 15, 17, 18 of 32 across consecutive
runs of the same command --- and the number that reached a figure was whichever
draw happened to be current.

A certificate whose value depends on when it was generated is not a
certificate. That is the article's own thesis turned on its author, so the fix
is here rather than a footnote: seeding is part of the protocol, the seed is
recorded in the certificate, and the RNGs that remain outside our control are
named rather than assumed away.
"""

from __future__ import annotations

import random

import numpy as np

__all__ = ["seed_all", "UNCONTROLLED"]

#: Generators known to ignore the seeds below, and why. A gate that depends on
#: one of these cannot promise reproducibility, and says so in its evidence.
UNCONTROLLED = {
    "albumentations": (
        "maintains its own generator, not derived from the Python, NumPy or "
        "PyTorch seeds; observed to change an operator's output under matched "
        "seeds"
    ),
}


def seed_all(seed: int) -> dict[str, object]:
    """Seed every generator we can reach; report what was reached.

    Returns a record for the certificate: which generators were seeded, and
    which known generators were not. Callers must not treat a successful return
    as a guarantee of bit-reproducibility --- it is a guarantee about the RNGs
    named in the record and nothing else.
    """
    seeded: list[str] = []

    random.seed(seed)
    seeded.append("python.random")

    np.random.seed(seed)
    seeded.append("numpy.random")

    try:  # torch is optional: the protocol runs on CPU with or without it
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        seeded.append("torch")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            seeded.append("torch.cuda")

    return {"seed": seed, "seeded": seeded, "uncontrolled": sorted(UNCONTROLLED)}
