"""Campaign P subject ``gfpgan``: GFPGAN's ``FFHQDegradationDataset``.

Blind face restoration (CVPR 2021, Wang et al.), audited at the loader the
shipped training configuration ``options/train_gfpgan_v1.yml`` names (line 11).
It reads only high-quality faces and manufactures the degraded input from each
one online, so like KAIR's blind SR loader and DDColor it declares no stored
input; unlike them, its degradation operators and their order are stated in the
publication, so the operator contract is the paper's and not a code reading.

Provenance of every contract value is given beside it: ``yml:N`` is a line of
``options/train_gfpgan_v1.yml``, ``ds:N`` a line of
``gfpgan/data/ffhq_degradation_dataset.py`` at the pinned commit, and
``paper`` is Section 4.1 of arXiv:2101.04061.
"""

KEY = "gfpgan"


def entry(h):
    return dict(
        module="adapters_gfpgan",
        cls="GfpganAdapter",
        root=h.Path("/home/laura/02ImproveData/zGFPGAN"),
        note="GFPGAN FFHQDegradationDataset; blind face restoration whose loader "
             "builds its input online from the target through the paper's "
             "degradation model, on the installed basicsr rather than a "
             "vendored one",
        # out_size: 512 (yml:21); the paper: "We resize all the images to
        # 512^2 during training". The loader stores out_size (ds:42) and uses
        # it only to mirror component boxes (ds:129-131).
        declared_shape=(512, 512),
        # paper Eq. (10): blur, then downsample by r, then additive Gaussian
        # noise, then JPEG; ds:166-182 apply them in that order every time.
        declared_policy="fixed",
        # The paper's four operators, as the adapter labels them. Colour
        # jitter, grey and hflip are recorded too but are probabilistic
        # presences, not orders, so they are not part of the order claim.
        ops={"blur", "resize", "gaussian_noise", "jpeg"},
        # No expected_transforms: neither the paper nor the yml states a count
        # of sharpening (or any other enhancement) applied to the target.
        num_frames=1,                 # single-image loader (ds:145-227)
        precomputed=False,            # no dataroot_lq option; lq built at ds:164-210
        has_target=True,              # 'gt' is returned (ds:227)
        clip_length=None,
        content=h.ContentContract(
            # mean = std = [0.5, 0.5, 0.5] (yml:19-20), applied to both
            # streams (ds:213-214) after values in [0, 1] (ds:150, ds:210):
            # [-1, 1]. Three channels: colour faces, BGR->RGB at ds:199.
            # Input extrema are not required: blur and noise of the fixture
            # need not reach them. Input channels are not required distinct:
            # gray_prob 0.01 (yml:35) replicates one channel (ds:191-193).
            lq=h.StreamContract(value_range=(-1.0, 1.0),
                                require_range_extrema=False, channels=3,
                                require_distinct_channels=False),
            # The target is only flipped and normalised (ds:156, ds:213);
            # gt_gray is not set in yml:8-52, so it stays colour.
            gt=h.StreamContract(value_range=(-1.0, 1.0),
                                require_range_extrema=True, channels=3,
                                require_distinct_channels=True)),
    )
