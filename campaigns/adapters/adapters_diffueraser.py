"""Adapter for DiffuEraser's ``TrainDataset`` --- a generative video-inpainting subject.

DiffuEraser (Li et al., 2025, arXiv:2501.10018,
https://github.com/lixiaowen-xw/DiffuEraser) is a diffusion model for video
inpainting: Stable Diffusion 1.5 with a BrushNet branch and, in its second
stage, an AnimateDiff motion module. Both training scripts build the same
loader, ``dataset/load_dataset.py:TrainDataset`` (``train_DiffuEraser_stage1.py:727``,
``train_DiffuEraser_stage2.py:723``). It reads whole videos listed in a CSV,
draws a random clip, generates a random moving mask, and returns the clip in
``[-1, 1]`` as ``pixel_values``, the masked clip as ``conditioning_pixel_values``
and the mask as ``masks``.

It is included because it is the generative member of the inpainting pair and
because of what it shares with ProPainter. ``dataset/utils.py`` is ProPainter's
``core/utils.py`` byte for byte except one trailing blank line, and the loader's
whole treatment --- the mask --- is ProPainter's
``create_random_shape_with_random_motion``. The loader around it is new: it
decodes videos with decord instead of reading frame folders, has no horizontal
flip (``GroupRandomHorizontalFlip`` is imported at line 10 and never called),
composes the masked input itself at the source resolution, and resizes the
short side instead of resizing to a fixed width and height. Two subjects
running one treatment inside two loaders separate the treatment from its
delivery.

Unlike the LaMa subject, the input is composed by the subject
(``load_dataset.py:129``), so the adapter delivers the subject's own pair:
``pixel_values`` as the target and ``conditioning_pixel_values`` as the input.

What the adapter supplies, recorded because adaptation effort is a response
variable:

* video files. The fixture writes PNG frames; the subject decodes video. They
  are encoded losslessly (FFV1) so the decoder cannot be mistaken for the
  loader, at 15 fps, the rate the loader resamples to (``load_dataset.py:93``),
  so its frame stride is 1 and the frames arrive in the order written;
* a tokenizer, which the constructor takes as an argument. The project's own
  SD-1.5 tokenizer is loaded the way its scripts load it
  (``train_DiffuEraser_stage1.py:610-615``), from local files;
* ``decord``, which ``load_dataset.py:6`` imports and the project's
  ``requirements.txt`` does not list. It is a real install, not a stub,
  because decoding is on the path every gate exercises.

Adaptation effort is a response variable of the study. Everything below the
header is that cost, and the subject is not modified to make a gate pass.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIFFUERASER_ROOT = Path("/home/laura/02ImproveData/zDiffuEraser/repo")
# The scripts name weights/stable-diffusion-v1-5 relative to the repository;
# in this checkout the weights folder sits beside it. Only the tokenizer files
# are read.
SD15 = DIFFUERASER_ROOT.parent / "weights" / "stable-diffusion-v1-5"

for base in (HERE, *HERE.parents):
    for cand in (base, base / "treatment_identity"):
        if (cand / "treatment_identity" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            break

from treatment_identity.adapter import LoaderSpec, Sample  # noqa: E402

#: --resolution=512 in both shipped scripts (train_DiffuEraser_stage1.sh:13,
#: train_DiffuEraser_stage2.sh:14), also the argparse default.
RESOLUTION = 512
#: The loader's own target rate: frame_stride = max(int(fps // 15), 1).
FIXTURE_FPS = 15
CAPTION = "a synthetic test pattern"


class _SeenTokenizer:
    """The project's tokenizer, recording the caption it is handed.

    The caption-dropout branch (``load_dataset.py:50``) replaces the caption
    with an empty string; what the tokenizer receives is the observation that
    the branch was taken, and nothing about the tokenization changes.
    """

    def __init__(self, tokenizer):
        self._tok, self.seen = tokenizer, []

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __call__(self, caption, **kw):
        self.seen.append(caption)
        return self._tok(caption, **kw)


class DiffuEraserAdapter:
    """Expose DiffuEraser's ``TrainDataset`` to the treatment-identity gates."""

    name = "diffueraser.TrainDataset"
    project = "DiffuEraser"
    url = "https://github.com/lixiaowen-xw/DiffuEraser"

    #: Set by :meth:`_restore_removed_method`. Reported in the certificate.
    environment_note: str | None = None

    def __init__(self, root: Path = DIFFUERASER_ROOT):
        self.root = Path(root)
        while str(self.root) in sys.path:
            sys.path.remove(str(self.root))
        sys.path.insert(0, str(self.root))
        self._restore_removed_method()
        from dataset.load_dataset import TrainDataset          # noqa: E402
        from transformers import AutoTokenizer                 # noqa: E402
        self._cls = TrainDataset
        self.tokenizer = _SeenTokenizer(AutoTokenizer.from_pretrained(
            str(SD15), subfolder="tokenizer", use_fast=False,
            local_files_only=True))
        self.last_branch_reached = False

    @classmethod
    def _restore_removed_method(cls) -> None:
        """Reinstate ``FigureCanvasAgg.tostring_rgb``, removed in Matplotlib 3.10.

        The mask generator rasterises a Bezier shape through a Matplotlib
        figure and reads it back with ``fig.canvas.tostring_rgb()``
        (``dataset/utils.py:295``). ``requirements.txt`` leaves Matplotlib
        unpinned, so on Python >= 3.10 a fresh install gets a release without
        the method and every sample raises ``AttributeError``. The replacement
        returns the same bytes, the RGB planes of the Agg buffer, and is
        applied in memory only.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        if hasattr(FigureCanvasAgg, "tostring_rgb"):
            return
        FigureCanvasAgg.tostring_rgb = (                     # type: ignore[attr-defined]
            lambda self: np.asarray(self.buffer_rgba())[..., :3].tobytes())
        cls.environment_note = (
            "matplotlib FigureCanvasAgg.tostring_rgb reinstated in memory over "
            "buffer_rgba; removed in Matplotlib 3.10, called by the mask "
            "generator, and Matplotlib is unpinned in requirements.txt")

    # -- adapter protocol ---------------------------------------------------
    def build(self, spec: LoaderSpec):
        """Encode each fixture clip as a video, list it in a CSV, build.

        Only ``gt_root`` is read: the subject has no stored-input branch, it
        manufactures the masked input from the clip.
        """
        import cv2

        gt_root = Path(spec.gt_root)
        stage = gt_root.parent / "diffueraser_stage"
        (stage / "video").mkdir(parents=True, exist_ok=True)
        rows = ["video_path,caption"]           # data/train/dataset1/metadata.csv:1
        for clip in sorted(p for p in gt_root.iterdir() if p.is_dir()):
            frames = sorted(clip.glob("*.png"))
            h, w = cv2.imread(str(frames[0])).shape[:2]
            # Re-encoded on every build: clip escalation rewrites a fixture in
            # place with more frames, and a cached video would hide that.
            out = stage / "video" / f"{clip.name}.avi"
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"FFV1"),
                                     FIXTURE_FPS, (w, h))
            for f in frames:
                writer.write(cv2.imread(str(f)))
            writer.release()
            rows.append(f"video/{clip.name}.avi,{CAPTION}")
        csv_file = stage / "metadata.csv"
        csv_file.write_text("\n".join(rows) + "\n")
        # The argparse fields TrainDataset reads, at their shipped values.
        args = types.SimpleNamespace(
            train_data_dir=str(stage), csv_file=[str(csv_file)],
            nframes=spec.num_frames, resolution=RESOLUTION,
            video_column="video_path", caption_column="caption",
            proportion_empty_prompts=0.0)
        self._ds = self._cls(args, self.tokenizer)
        return self._ds

    def sample(self, loader, index: int) -> Sample:
        item = loader[index % len(loader)]
        return Sample(gt=_thwc(item["pixel_values"]),
                      lq=_thwc(item["conditioning_pixel_values"]),
                      frame_ids=None,     # the clip's frame_indices stay local
                      extra={"mask": _thwc(item["masks"])})

    def __len__(self) -> int:
        return len(self._ds)

    # -- settings sweep -----------------------------------------------------
    #: The one treatment switch the training scripts expose for this loader:
    #: caption dropout (train_DiffuEraser_stage1.py:452, default 0, set by
    #: neither shipped script). The mask type, clip reversal and resizing are
    #: hard-coded and have no switch to sweep.
    def settings(self):
        return {"proportion_empty_prompts": (0.0, 1.0)}

    def deliver_under(self, workdir: Path, name: str, value):
        """Deliver the token ids of one item with one option set.

        Returns them with whether the dropout branch was taken, read off the
        caption the tokenizer received.
        """
        from treatment_identity.fixtures import make_pair

        fixture = Path(workdir) / "fixture"
        if not (fixture / "GT").exists():
            make_pair(fixture, n_frames=32, shape=(128, 128), gt_kind="signature")
        ds = self.build(LoaderSpec(gt_root=fixture / "GT", num_frames=10))
        setattr(ds.args, name, value)
        self.tokenizer.seen.clear()
        self.last_branch_reached = False
        item = ds[0]
        self.last_branch_reached = self.tokenizer.seen[-1] == ""
        return np.asarray(item["input_ids"]), self.last_branch_reached


def _thwc(t) -> np.ndarray:
    """Torch (T, C, H, W) -> numpy (T, H, W, C), values untouched."""
    return np.transpose(t.detach().cpu().numpy(), (0, 2, 3, 1))
