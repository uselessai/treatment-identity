"""Campaign P entry for DART (TytanMikJas/DART-Degradation-Aware-Recurrent-
Transformer), under its training configuration ``configs/train.yaml``, with
AbsoluteDegradation v1.0.0 for the online degradation. DART runs in its own
environment, so this subject is profiled with that environment's Python."""

KEY = "dart"


def entry(h):
    return dict(
        module="adapters_dart",
        cls="DartAdapter",
        root=h.Path("/home/laura/02ImproveData/zDART/repo"),
        note="DART VideoFrameSequenceDataset; old-film restoration whose loader "
             "degrades clean clips on the fly with AbsoluteDegradation and also "
             "delivers a mask of the defects",
        # configs/train.yaml: data.crop_size 256, data.num_frames 7
        declared_shape=(256, 256),
        num_frames=7,
        precomputed=False,
        content=h.rgb_contract((-1.0, 1.0), lq_extrema=False),
    )
