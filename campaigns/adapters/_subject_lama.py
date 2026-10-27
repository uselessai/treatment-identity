"""Campaign P entry for LaMa (advimman/lama), audited under the CelebA-HQ
training configuration ``configs/training/lama-fourier-celeba.yaml``, whose
data is ``configs/training/data/abl-04-256-mh-dist-celeba.yaml``. Line
references below are to the subject at commit 786f5936b27f."""

KEY = "lama"


def entry(h):
    return dict(
        module="adapters_lama",
        cls="LamaAdapter",
        root=h.Path("/home/laura/02ImproveData/zLaMa"),
        note="LaMa InpaintingTrainDataset (CelebA-HQ data config); an "
             "inpainting loader whose treatment is a mask it generates per "
             "item from a weighted mixture of mask kinds, not a degradation",
        # abl-04-256-mh-dist-celeba.yaml:8 'out_size: 256'.
        declared_shape=(256, 256),
        # The input is generated online from the image: the loader reads one
        # directory of images (datasets.py:27) and draws the mask itself
        # (datasets.py:42). There is no stored-input branch to declare.
        precomputed=False,
        # Single-image loader: one image per item (datasets.py:36-39).
        num_frames=1,
        # The two kinds with non-zero weight: irregular_proba: 1 and
        # box_proba: 1, segm_proba: 0 (abl-04-256-mh-dist-celeba.yaml:11, 19,
        # 27), re-normalised and drawn once per item (masks.py:305-310).
        ops={"irregular", "rectangle"},
        # The configuration declares a random per-item CHOICE of kind, not an
        # order; the gate's vocabulary has only "fixed" and
        # "random_permutation", and the latter is the one whose operational
        # test (the realised operator sequence varies across draws) matches the
        # declaration. "fixed" would contradict the configuration.
        declared_policy="random_permutation",
        # No sharpening or other counted target transform is stated anywhere,
        # so expected_transforms is deliberately absent.
        # Value range [0, 1]: 'transform_variant: no_augs'
        # (abl-04-256-mh-dist-celeba.yaml:29) is A.Compose([A.ToFloat()])
        # (datasets.py:197-200), uint8 / 255, so the target keeps the fixture
        # anchors. Three RGB channels: cv2.cvtColor BGR2RGB (datasets.py:38).
        # The input image * (1 - mask) (trainers/default.py:59) zeroes the hole
        # by design, so its upper anchor is not part of any declaration.
        content=h.rgb_contract((0.0, 1.0), lq_extrema=False),
    )
