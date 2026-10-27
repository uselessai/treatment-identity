"""Study 3 subject: ProPainter ``TrainDataset`` (video inpainting).

ProPainter (ICCV 2023, https://github.com/sczhou/ProPainter) trains on clips
whose input is manufactured online: a moving random mask is drawn per item and
the trainer blanks the frames under it. It is included as the only video
inpainting loader of the study, and as a window that is not one contiguous run
(``num_local_frames`` consecutive frames plus ``num_ref_frames`` references).

Every expected value below names where it was read. Line numbers refer to the
pinned commit e870e79321c3.
"""

KEY = "propainter"


def entry(h):
    return dict(
        module="adapters_propainter",
        cls="ProPainterAdapter",
        root=h.Path("/home/laura/02ImproveData/zProPainter"),
        note="ProPainter TrainDataset; video inpainting with an online moving "
             "mask, a local-plus-reference window and a full-frame resize",
        # (H, W) = (h, w) of configs/train_propainter.json:8-9; README.md:224
        # also tells users to resize training frames to 432x240.
        declared_shape=(240, 432),
        # Source-code reading: mask generation, then the flip, in straight-line
        # code (core/dataset.py:73, :123). Nothing declares a random order.
        declared_policy="fixed",
        # No count of target transforms is stated by the configuration, the
        # README or the loader, so ``expected_transforms`` is omitted.
        # num_local_frames + num_ref_frames = 10 + 6
        # (configs/train_propainter.json:10-11; core/dataset.py:61-68).
        num_frames=16,
        # The input is generated: the mask is drawn per item
        # (core/dataset.py:73) and TrainDataset reads no stored mask or masked
        # frame; the configuration names no such path (train_propainter.json:4-13).
        precomputed=False,
        # Clips of 16 frames or fewer are dropped silently (core/dataset.py:45);
        # neither the configuration nor the README states a minimum length, so
        # none is declared here.
        clip_length=None,
        # Frames in [-1, 1] (core/dataset.py:126, comment at :134), RGB after
        # cv2.COLOR_BGR2RGB (:90). The input is frames * (1 - masks) as the
        # trainer composes it (core/trainer.py:363), so it shares that range.
        content=h.rgb_contract((-1.0, 1.0)),
    )
