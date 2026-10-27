"""Campaign P subject: E2FGVI ``TrainDataset`` (flow-guided video inpainting).

E2FGVI (Li et al., CVPR 2022, https://github.com/MCG-NKU/E2FGVI) is included
because its temporal sampling is not one block of consecutive frames: each item
is five consecutive local frames plus three non-local reference frames drawn at
random from the rest of the clip. It is also a second inpainting subject after
LaMa, and the first whose mask treatment spans a clip.

Every expected value below is what the project's own configuration, README or
publication states, with its source; where only the code says it, that is said.
"Paper" is Li et al., "Towards An End-to-End Framework for Flow-Guided Video
Inpainting", arXiv:2204.02663, Appendix A ("Training details") unless noted.
"""

KEY = "e2fgvi"


def entry(h):
    return dict(
        module="adapters_e2fgvi",
        cls="E2fgviAdapter",
        root=h.Path("/home/laura/02ImproveData/zE2FGVI"),
        note="E2FGVI TrainDataset; video inpainting whose window is 5 "
             "consecutive local frames plus 3 randomly sampled reference "
             "frames, with an online clip mask as its treatment",
        # (H, W) = (240, 432): configs/train_e2fgvi.json:7-8 ("w": 432,
        # "h": 240); paper: "we resize all frames from videos into 432x240 for
        # training".
        declared_shape=(240, 432),
        # Paper: "we resize all frames ... into 432x240"; README.md:167:
        # "E2FGVI always rescales the input video to a fixed resolution
        # (432x240)". A fixed output size whatever the source proportions is
        # what this vocabulary calls a reshape.
        declared_content="reshape",
        # 5 + 3 = 8 frames per item: configs/train_e2fgvi.json:9-10
        # (num_local_frames 5, num_ref_frames 3); paper: "the numbers of local
        # (Tl) and non-local frames (Tnl) are 5 and 3 ... Local frames are
        # continuous clips, while non-local frames are randomly sampled".
        num_frames=8,
        # Source reading only (core/dataset.py:55-76): mask generation, then
        # frame selection, read and resize, then one random horizontal flip,
        # always in that order. Neither paper nor config states an order, and
        # no step is composed in a random sequence.
        declared_policy="fixed",
        # The training loader has no stored-input or stored-mask branch: masks
        # are always generated (core/dataset.py:55); stored masks are read only
        # by TestDataset (core/dataset.py:117).
        precomputed=False,
        # The target is the clip itself (core/trainer.py:347, loss against
        # ``frames``). No source states a count of target-side transforms, so
        # ``expected_transforms`` is deliberately absent.
        has_target=True,
        # [-1, 1]: source code only, core/dataset.py:77 (``* 2.0 - 1.0``); the
        # paper does not state a range. 3 channels: core/dataset.py:70
        # (``convert('RGB')``). The input is frames * (1 - masks)
        # (core/trainer.py:304): holes are set to 0 and may cover both range
        # anchors, so only the target is required to reach the extrema.
        content=h.rgb_contract((-1.0, 1.0), lq_extrema=False),
    )
