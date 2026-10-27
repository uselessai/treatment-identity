"""Campaign P subject: VRT, trained through KAIR's video loader.

VRT (https://github.com/JingyunLiang/VRT, pinned ed9498a9de21) ships no
training loader; its README points to KAIR (README.md:154), and KAIR
(fc1732f4a451) ships the VRT training configurations. The class is the one
subject ``kair_video`` adapts. It is included because a published model's own
configuration is a different contract for the same class: scale 4 against a
pre-rendered bicubic input, 256-pixel target patches, six frames, flips and
rotations on. The harness pins ``root`` only, so the KAIR commit that actually
executes is stated in the note.

Every value below is what VRT's configuration or code requests. Line numbers
refer to KAIR ``options/vrt/001_train_vrt_videosr_bi_reds_6frames.json``
unless another file is named.
"""

KEY = "vrt"


def entry(h):
    return dict(
        module="adapters_vrt",
        cls="VrtTrainAdapter",
        root=h.Path("/home/laura/02ImproveData/zVRT"),
        note="VRT video SR x4 on REDS (6 frames); VRT ships no training loader "
             "and trains through KAIR fc1732f4a451 VideoRecurrentTrainDataset "
             "under KAIR options/vrt/001 -- the kair_video class under the "
             "configuration a published model used",
        # gt_size 256 (:31); the network's LQ patch of 64 (:55, img_size
        # [6,64,64]) is the same request at scale 4 (:9).
        declared_shape=(256, 256),
        # The input is opened, not generated: dataroot_lq is the offline
        # bicubic render (:23; VRT README.md:145 "REDS sharp & sharp_bicubic"),
        # read at dataset_video_train.py:155-156.
        precomputed=True,
        # num_frame (:30).
        num_frames=6,
        # No clip_length: the loader takes each clip's length from the
        # meta-info file (dataset_video_train.py:62-67) and states no minimum.
        # No expected_transforms and no declared_policy: neither the
        # configuration nor the README states a target-transform count or a
        # degradation-operator order; the input is a stored render.
        # Value range: no configuration key states one. Source reading:
        # imfrombytes(float32=True) divides by 255 (utils/utils_video.py:490-491,
        # called at dataset_video_train.py:156,160), and img2tensor keeps it.
        # Channels: n_channels 3 (:10), broadcast by utils/utils_option.py:59.
        content=h.rgb_contract((0.0, 1.0)),
    )
