"""Campaign P subject: DiffuEraser ``TrainDataset`` (diffusion video inpainting).

DiffuEraser (https://github.com/lixiaowen-xw/DiffuEraser, pinned at 8e6f279ac753)
trains a Stable Diffusion 1.5 video inpainter from one loader shared by both
training stages. It is included as the generative member of the inpainting
pair: its mask generator is ProPainter's, copied verbatim into
``dataset/utils.py``, while the loader around it is its own. The input is
manufactured online --- the clip with a random moving hole --- so the subject
also exercises the pre-computed-input branch from the side that declares none.

Every expected value below states where it was read. Nothing here was set
after seeing a gate's output.
"""

KEY = "diffueraser"


def entry(h):
    return dict(
        module="adapters_diffueraser",
        cls="DiffuEraserAdapter",
        root=h.Path("/home/laura/02ImproveData/zDiffuEraser/repo"),
        note="DiffuEraser TrainDataset; diffusion video inpainting whose mask "
             "generator is ProPainter's, copied verbatim, inside a loader of "
             "its own that decodes videos and builds the masked input itself",
        # --resolution=512 (train_DiffuEraser_stage1.sh:13), whose help text
        # says every training image "will be resized to this resolution"
        # (train_DiffuEraser_stage1.py:257-259). The gate probes a square
        # source, so it cannot see what the loader does to a wide one.
        declared_shape=(512, 512),
        # Source reading: the sequence is fixed in code (load_dataset.py:113-145);
        # only branch choices are random, not the order of operations.
        declared_policy="fixed",
        # No count of target-side transforms is stated by the scripts or the
        # publication, so expected_transforms is deliberately absent.
        # --nframes=10 in the stage-1 script (train_DiffuEraser_stage1.sh:14);
        # stage 2 uses 22 (train_DiffuEraser_stage2.sh:15), argparse default 24.
        num_frames=10,
        # The masked input is composed from the clip (load_dataset.py:129);
        # there is no stored-input branch to honour.
        precomputed=False,
        # pixel_values is what the diffusion loss reconstructs (encoded to
        # latents at train_DiffuEraser_stage1.py:852).
        has_target=True,
        # Both streams pass ToTensor + Normalize([0.5],[0.5]), commented
        # "[-1, 1]" (load_dataset.py:28-32, 148-149); three RGB channels from
        # decord (load_dataset.py:107). Extrema are required on both streams,
        # as for the other RGB subjects: they are what separates [-1, 1] from
        # a stream left in [0, 1]. The input's hole is set to the low anchor
        # (load_dataset.py:129) and does not cover the whole frame.
        content=h.rgb_contract((-1.0, 1.0)),
        # No source states a crop, rescale or reshape in the gate's sense: the
        # help text says "resized" without saying whether proportions are kept.
    )
