"""Connect DDC to a loader without writing an adapter.

The wizard runs the project's own training command, stops it when the training
``DataLoader`` is created, and keeps the dataset that the project built from
its own configuration. The data the dataset reads are replaced at read time:
every image file under a data folder is served from the current DDC fixture,
so the project's paths, file lists, option names and framework need no
translation, and none of the user's own pixels are read. What it cannot work
out, it asks; the answers are kept in a JSON file so that the next run asks
nothing.

    python -m treatment_identity.wizard --cwd <project> --answers ddc.json \\
        -- python train.py -opt options/train.yml

A question without an answer is printed with what the wizard detected, and the
run stops with exit status 2 until the answer is added.
"""
from __future__ import annotations

import argparse
import builtins
import contextlib
import io
import json
import os
import re
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np

from .adapter import AdapterError, CallRecorder, LoaderSpec, Sample

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
TARGET_NAMES = ("gt", "hq", "h", "clean", "target", "sharp", "groundtruth", "y")
INPUT_NAMES = ("lq", "lr", "l", "input", "degraded", "noisy", "blur", "x", "inp", "conditioning")
MASK_NAMES = ("mask", "masks", "hole", "holes", "m")


#: The environment before the project's code ran in this process; runs with a
#: changed configuration start from it, not from what the project set.
ENVIRONMENT = dict(os.environ)
#: File lists the run reads its data from (see ``standin_lists``).
LISTS: list[str] = []

#: Folders and files the wizard created; listed at the end, never removed.
CREATED: list[Path] = []
#: Existing data folders without images, to which stand-in files were added.
STANDIN_IN_EXISTING: list[tuple[Path, set]] = []


class Question(Exception):
    """Raised when the wizard needs an answer it does not have."""

    def __init__(self, key: str, text: str, suggestion=None):
        super().__init__(text)
        self.key, self.text, self.suggestion = key, text, suggestion


class Captured(Exception):
    """Raised inside the project when its training DataLoader is created."""

    def __init__(self, dataset):
        super().__init__("dataset captured")
        self.dataset = dataset


# --------------------------------------------------------------- answers
class Answers:
    def __init__(self, path: Path | None):
        self.path = path
        self.data = json.loads(path.read_text()) if path and path.exists() else {}
        self.detected: list[str] = []

    def get(self, key, text, suggestion=None, sure=False):
        """The recorded answer; else the detected value when ``sure``; else ask.

        A value used without asking is still written to the answers file,
        under its key, so it can be read and corrected there.
        """
        if key not in self.data and sure and suggestion is not None:
            self.data[key] = suggestion
            self.detected.append(key)
            self.save()
            return suggestion
        if key in self.data:
            # "detected" accepts what the wizard proposed
            if self.data[key] == "detected" and suggestion is not None:
                return suggestion
            return self.data[key]
        if sys.stdin.isatty():
            shown = f" [{json.dumps(suggestion)}]" if suggestion is not None else ""
            reply = input(f"{text}{shown}: ").strip()
            try:
                value = json.loads(reply) if reply else suggestion
            except json.JSONDecodeError:
                value = reply                      # a plain word, such as an entry name
            self.data[key] = value
            self.save()
            return value
        raise Question(key, text, suggestion)

    def save(self):
        if self.path:
            self.path.write_text(json.dumps(self.data, indent=2) + "\n")


# ----------------------------------------------------- virtual image data
def _frame_position(path: Path) -> int:
    """Position of a frame within its clip, from the last number in its name."""
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else 0


class _Frames:
    def __init__(self, array):
        self._array = array

    def asnumpy(self):
        return self._array


class _ServedVideo:
    """What ``decord.VideoReader`` returns, for a video served from a fixture."""

    def __init__(self, frames: list[np.ndarray], fps: float):
        self._frames, self._fps = np.stack(frames), fps

    def __len__(self):
        return len(self._frames)

    def get_avg_fps(self):
        return self._fps

    def get_batch(self, indices):
        return _Frames(self._frames[list(indices)])

    def __getitem__(self, index):
        return _Frames(self._frames[index])


class VirtualData:
    """Serve every image read under the data folders from the current fixture."""

    def __init__(self, roots: dict[str, str]):
        # folder -> stream ("gt" or "lq"); longest folder first, so a nested
        # folder wins over the one that contains it
        self.roots = sorted(((str(Path(r).resolve()), s) for r, s in roots.items()),
                            key=lambda t: -len(t[0]))
        self.frames: dict[str, list[np.ndarray]] = {}
        self.trace: list[str] = []
        self.min_size = 0
        # padding to min_size is used only once a loader fails without it
        self.pad = False
        self._orig = {}
        # other files the run reads, served with this text instead
        self.served: dict[str, str] = {}
        # frame rate of a served video; the fixture is a sequence of frames
        self.fps = 25.0

    def stream_of(self, path) -> str | None:
        try:
            p = str(Path(os.fspath(path)).resolve())
        except (TypeError, ValueError):
            return None
        if Path(p).suffix.lower() not in IMAGE_EXT:
            return None
        for root, stream in self.roots:
            if p == root or p.startswith(root + os.sep):
                return stream
        return None

    def load(self, stream: str, folder: Path, scale: int = 1, multiple: int = 1):
        """Use the frames of the first clip of a DDC fixture folder.

        A stored input for super-resolution is ``scale`` times smaller than its
        target; it is reduced by bicubic interpolation, as such inputs are made.
        """
        clips = sorted(p for p in Path(folder).iterdir() if p.is_dir()) if Path(folder).is_dir() else []
        files = sorted((clips[0] if clips else Path(folder)).glob("*.png"))
        frames = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in files]
        # A fixture smaller than the patch the loader crops is repeated to that
        # size, which keeps its values and the scale of its content (an
        # enlargement would be measured as the loader's).
        if frames:
            h, w = frames[0].shape[:2]
            size = self.min_size if self.pad else 0
            ph, pw = max(0, size - h), max(0, size - w)
            if ph or pw:
                pad = ((0, ph), (0, pw)) + ((0, 0),) * (frames[0].ndim - 2)
                frames = [np.pad(f, pad, mode="wrap") for f in frames]
        if multiple > 1 or scale > 1:
            # Super-resolution pairs need a target whose sides are multiples of
            # the scale, as the scripts that make such pairs crop it.
            m = max(multiple, scale)
            frames = [f[:f.shape[0] - f.shape[0] % m, :f.shape[1] - f.shape[1] % m]
                      for f in frames]
        if scale > 1:
            frames = [cv2.resize(f, (f.shape[1] // scale, f.shape[0] // scale),
                                 interpolation=cv2.INTER_CUBIC) for f in frames]
        self.frames[stream] = frames

    def _stream_under(self, path, suffixes) -> str | None:
        try:
            full = str(Path(os.fspath(path)).resolve())
        except (TypeError, ValueError):
            return None
        if Path(full).suffix.lower() not in suffixes:
            return None
        stream = next((s for r, s in self.roots if full.startswith(r + os.sep)), None)
        return stream if stream is not None and self.frames.get(stream) else None

    def video_for(self, path) -> list[np.ndarray] | None:
        """A video file under the data folders, as the current fixture's frames
        (RGB, as video decoders return them), repeated to 100 frames."""
        stream = self._stream_under(path, VIDEO_EXT)
        if stream is None:
            return None
        self.trace.append(os.fspath(path))
        frames = self.frames[stream]
        return [cv2.cvtColor(frames[i % len(frames)], cv2.COLOR_BGR2RGB)
                for i in range(max(len(frames), 100))]

    def archive_for(self, path) -> bytes | None:
        """A clip stored as a ZIP of frames, rebuilt from the current fixture."""
        p = os.fspath(path)
        if not p.lower().endswith(".zip"):
            return None
        try:
            full = str(Path(p).resolve())
        except (TypeError, ValueError):
            return None
        stream = next((s for r, s in self.roots if full.startswith(r + os.sep)), None)
        if stream is None or not self.frames.get(stream):
            return None
        import zipfile
        buf = io.BytesIO()
        frames = self.frames[stream]
        # as many frames as a fixture clip has, whatever is being served now;
        # positions past the fixture repeat it, as frame_for does
        with zipfile.ZipFile(buf, "w") as z:
            for i in range(max(len(frames), 100)):
                ok, enc = cv2.imencode(".png", frames[i % len(frames)])
                z.writestr(f"{i:05d}.jpg", enc.tobytes())
        self.trace.append(p)
        return buf.getvalue()

    def frame_for(self, path) -> np.ndarray | None:
        stream = self.stream_of(path)
        if stream is None or stream not in self.frames or not self.frames[stream]:
            return None
        self.trace.append(os.fspath(path))
        frames = self.frames[stream]
        return frames[_frame_position(Path(os.fspath(path))) % len(frames)]

    @contextlib.contextmanager
    def active(self):
        orig_open, orig_imread = builtins.open, cv2.imread
        data = self

        def fake_open(file, mode="r", *a, **kw):
            if isinstance(file, (str, os.PathLike)) and "r" in mode and data.served:
                text = data.served.get(str(Path(os.fspath(file)).resolve()))
                if text is not None:
                    f = io.BytesIO(text.encode()) if "b" in mode else io.StringIO(text)
                    f.name = os.fspath(file)        # readers that report the file's name
                    return f
            if isinstance(file, (str, bytes, os.PathLike)) and "r" in mode and "b" in mode:
                archive = data.archive_for(file)
                if archive is not None:
                    return io.BytesIO(archive)
            if isinstance(file, (str, bytes, os.PathLike)) and "r" in mode:
                frame = data.frame_for(file)
                if frame is not None:
                    ok, buf = cv2.imencode(".png", frame)
                    raw = buf.tobytes()
                    return io.BytesIO(raw) if "b" in mode else io.StringIO(raw.decode("latin-1"))
            return orig_open(file, mode, *a, **kw)

        def fake_imread(filename, flags=cv2.IMREAD_COLOR):
            frame = data.frame_for(filename)
            if frame is None:
                return orig_imread(filename, flags)
            ok, buf = cv2.imencode(".png", frame)
            return cv2.imdecode(buf, flags)

        # Videos decoded by decord are served too. A module that imported the
        # reader while this was active keeps the served one, which falls back
        # to decord for any other file.
        import importlib.util
        decord = importlib.import_module("decord") if importlib.util.find_spec("decord") else None
        if decord is not None:
            orig_reader = decord.VideoReader

            def reader(uri, *a, **kw):
                frames = data.video_for(uri)
                return _ServedVideo(frames, data.fps) if frames is not None \
                    else orig_reader(uri, *a, **kw)

            decord.VideoReader = reader
        builtins.open, io.open, cv2.imread = fake_open, fake_open, fake_imread
        try:
            yield self
        finally:
            builtins.open, io.open, cv2.imread = orig_open, orig_open, orig_imread
            if decord is not None:
                decord.VideoReader = orig_reader


# ------------------------------------------------------------- capturing
@contextlib.contextmanager
def _in_project(cwd: Path, argv: list[str]):
    old_cwd, old_argv, old_path = Path.cwd(), sys.argv[:], sys.path[:]
    os.chdir(cwd)
    sys.argv = argv[:]
    sys.path[:0] = [str(cwd), str((cwd / argv[0]).resolve().parent)]
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.argv, sys.path[:] = old_argv, old_path


def _folders_named(dataset, cwd: Path) -> set[str]:
    """The folders a dataset's own attributes name, resolved."""
    out = set()
    for value in vars(dataset).values() if hasattr(dataset, "__dict__") else []:
        if isinstance(value, (str, os.PathLike)) and str(value) and len(str(value)) < 4096:
            path = Path(os.fspath(value))
            out.add(str((cwd / path).resolve() if not path.is_absolute() else path.resolve()))
    return out


def capture_dataset(cwd: Path, command: list[str], data: VirtualData | None = None,
                    setup: list[str] | None = None, skip_roots: dict | None = None):
    """Run the training command until its training DataLoader exists; return the dataset.

    A loader whose dataset names only validation or test folders (``skip_roots``)
    is built as usual, and the run continues.
    """
    import torch.utils.data as tud

    if command and Path(command[0]).name.startswith("python"):
        command = command[1:]
    # Hydra replaces an error inside the run by a one-line message otherwise.
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    original_init = tud.DataLoader.__init__
    skip = {str(Path(r).resolve()) for r in skip_roots or {}}
    train = {r for r, _ in data.roots} - skip if data else set()

    def stop(self, dataset, *a, **kw):
        named = _folders_named(dataset, cwd)
        names = lambda roots: any(n == r or n.startswith(r + os.sep)       # noqa: E731
                                  for n in named for r in roots)
        if names(skip) and not names(train):
            return original_init(self, dataset, *a, **kw)
        captured.append(dataset)
        raise Captured(dataset)

    # kept here too: a script that catches every exception and exits would
    # otherwise lose it
    captured: list = []

    tud.DataLoader.__init__ = stop
    # Scripts that start one process per GPU through torch.multiprocessing
    # start none on a CPU; rank 0 is run here, in this process, instead.
    import torch.multiprocessing as tmp
    original_spawn = tmp.spawn

    def spawn_rank0(fn, args=(), nprocs=1, *a, **kw):
        fn(0, *args)

    tmp.spawn = spawn_rank0
    # Moving the model to a GPU before the loader exists is common; on a CPU
    # those calls do nothing here, since the run stops before any training.
    import torch
    cuda_orig = (torch.nn.Module.cuda, torch.Tensor.cuda, torch.cuda.set_device)
    if not torch.cuda.is_available():
        torch.nn.Module.cuda = lambda self, *a, **kw: self
        torch.Tensor.cuda = lambda self, *a, **kw: self
        torch.cuda.set_device = lambda *a, **kw: None
    ctx = data.active() if data else contextlib.nullcontext()
    try:
        with _in_project(cwd, command), ctx:
            # Environment preparation the project would have had installed;
            # given by the user, never guessed.
            for line in setup or []:
                exec(line, {"__name__": "__ddc_setup__"})
            try:
                if command[0] == "-m":
                    runpy.run_module(command[1], run_name="__main__", alter_sys=True)
                else:
                    runpy.run_path(command[0], run_name="__main__")
            except Captured as caught:
                return caught.dataset
            except BaseException as exc:
                if captured:
                    return captured[0]
                if not isinstance(exc, SystemExit):
                    raise
                raise RuntimeError(f"the training command exited ({exc.code}) "
                                   "before creating a DataLoader") from exc
    finally:
        tud.DataLoader.__init__ = original_init
        tmp.spawn = original_spawn
        torch.nn.Module.cuda, torch.Tensor.cuda, torch.cuda.set_device = cuda_orig
    raise RuntimeError("the training command finished without creating a DataLoader")


# ------------------------------------------------------------- the adapter
def _arrays(item):
    """(name, array) for every image-like entry of a dataset item."""
    import torch
    entries = item.items() if isinstance(item, dict) else enumerate(item) \
        if isinstance(item, (tuple, list)) else [(0, item)]
    out = []
    for name, value in entries:
        # pictures held by an object one level down (MMagic packs the target
        # into a "data_samples" object), named by their path
        inner = [(f"{name}.{k}", v) for k, v in vars(value).items()] \
            if hasattr(value, "__dict__") and not isinstance(value, torch.Tensor) else []
        for n, v in [(name, value)] + inner:
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            if isinstance(v, np.ndarray) and v.ndim >= 2 and v.size > 16:
                out.append((n, v))
    return out


def _guess(names, words):
    for n in names:
        key = re.sub(r"[^a-z]", "", str(n).rsplit(".", 1)[-1].lower())
        if key in words or any(key.startswith(w) and len(w) > 1 for w in words):
            return n
    return None


def _forget_open_archives(cwd: Path):
    """Empty the project's own caches of open ZIP archives, which would keep
    serving the frames of an earlier fixture."""
    import zipfile
    root = str(cwd.resolve())
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None) or ""
        # the project's own code, not packages installed inside it (a .venv)
        if not str(Path(f).resolve()).startswith(root + os.sep) \
                or "site-packages" in f or "dist-packages" in f:
            continue
        holders = [vars(module)] + [vars(c) for c in vars(module).values() if isinstance(c, type)]
        for holder in holders:
            for value in list(holder.values()):
                if type(value) is dict and any(isinstance(v, zipfile.ZipFile)
                                               for v in list(value.values())):
                    value.clear()


class WizardAdapter:
    """An adapter made from a captured dataset and the virtual data."""

    def __init__(self, dataset, data: VirtualData, answers: Answers, name="wizard",
                 scale: int = 1, cwd: Path | None = None):
        self.dataset, self.data, self.answers, self.name = dataset, data, answers, name
        self.scale, self.cwd = scale, cwd
        self.recorder = CallRecorder()
        for target in answers.data.get("observe", []):     # "module:function"
            module, _, attr = target.partition(":")
            owner = sys.modules.get(module)
            if owner is not None:
                self.recorder.wrap(owner, attr)
        with self.data.active():
            pass

    def keys(self, item):
        names = [n for n, _ in _arrays(item)]
        g, i = _guess(names, TARGET_NAMES), _guess(names, INPUT_NAMES)
        mask = _guess(names, MASK_NAMES)
        if i is None and mask is not None:
            # inpainting: the model sees the target with the mask's holes removed
            i = {"masked_by": mask}
        # with the input and the mask known, a single remaining entry is the target
        rest = [n for n in names if n != mask and n != i]
        if g is None and i is not None and len(rest) == 1:
            g = rest[0]
        sure = g is not None and i is not None and g != i
        gt = self.answers.get("target_key", f"Which entry of a dataset item is the target? {names}",
                              g, sure=sure)
        lq = self.answers.get(
            "input_key", f"Which entry of a dataset item is the model input? {names} "
            "(for inpainting, {\"masked_by\": <mask entry>}: the target with the "
            "mask's non-zero pixels removed)", i, sure=sure)
        return lq, gt

    def build(self, spec: LoaderSpec):
        self._spec = spec
        self.data.load("gt", Path(spec.gt_root), multiple=self.scale)
        self.data.load("lq", Path(spec.lq_root or spec.gt_root), self.scale)
        if self.cwd is not None:
            _forget_open_archives(self.cwd)
        self._reads_at_build = len(self.data.trace)
        return self.dataset

    def sample(self, loader, index: int) -> Sample:
        # Fixtures are served at their own size, so that whatever the loader
        # does to their size is seen. One that is smaller than the patch the
        # loader crops makes it fail; from then on fixtures are repeated to
        # that size, which keeps the scale of their content.
        try:
            with self.data.active():
                item = loader[index % len(loader)]
        except Exception:                                   # noqa: BLE001
            if self.data.pad or not self.data.min_size or getattr(self, "_spec", None) is None:
                raise
            self.data.pad = True
            self.build(self._spec)
            with self.data.active():
                item = loader[index % len(loader)]
        if len(self.data.trace) == getattr(self, "_reads_at_build", -1):
            # The fixtures reach a dataset only through what it reads; one that
            # read nothing since the fixture was loaded (it loaded its data when
            # it was built) delivers those data, not the fixture. A dataset that
            # keeps an archive open after reading it once still read it.
            raise AdapterError("the dataset read no data file to deliver this item: it loaded "
                               "its data when it was built, so the fixtures do not reach it")
        lq_key, gt_key = self.keys(item)
        arrays = dict(_arrays(item))
        gt = arrays.get(gt_key)
        if isinstance(lq_key, dict):
            mask = arrays[lq_key["masked_by"]]
            return Sample(lq=gt * (mask == 0), gt=gt, frame_ids=None)
        return Sample(lq=arrays.get(lq_key), gt=gt, frame_ids=None)

    def __len__(self):
        return len(self.dataset)


# ------------------------------------------------------------- discovery
CONFIG_EXT = {".yml", ".yaml", ".json"}


def _strip_comment(line: str) -> str:
    """A line of JSON without its // comment; slashes inside strings are kept."""
    quoted = False
    for i, ch in enumerate(line):
        if ch == '"' and (i == 0 or line[i - 1] != "\\"):
            quoted = not quoted
        elif not quoted and line.startswith("//", i):
            return line[:i]
    return line


#: Configuration files as this run supplies them (see ``served_settings``):
#: the wizard reads them as the project will.
SERVED_CONFIG: dict[str, str] = {}


def _load_config(path: Path):
    """A YAML or JSON configuration as nested data; JSON may carry // comments."""
    text = SERVED_CONFIG.get(str(Path(path).resolve())) or path.read_text()
    if path.suffix.lower() == ".json":
        text = "\n".join(_strip_comment(line) for line in text.splitlines())
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    import yaml
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None


def _walk(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{prefix}.{i}")
    else:
        yield prefix, obj


def _script(cwd: Path, command: list[str]) -> Path | None:
    args = command[1:] if command and Path(command[0]).name.startswith("python") else command
    if not args:
        return None
    if args[0] == "-m" and len(args) > 1:
        cand = cwd / (args[1].replace(".", "/") + ".py")
        return cand if cand.is_file() else None
    return (cwd / args[0]) if (cwd / args[0]).is_file() else None


def _config_files(cwd: Path, command: list[str]) -> list[Path]:
    """Configuration files the run reads: named on the command line, named as a
    string in the entry script, or composed by Hydra from the script's decorator."""
    files = [cwd / a for a in command if (cwd / a).suffix.lower() in CONFIG_EXT
             and (cwd / a).is_file()]
    script = _script(cwd, command)
    if script is not None:
        src = script.read_text(errors="ignore")
        for lit in re.findall(r"""["']([\w./-]+\.(?:ya?ml|json))["']""", src):
            for base in (cwd, script.parent):
                if (base / lit).is_file():
                    files.append(base / lit)
                    break
        # a folder the script names, and a file in it named by a value of the
        # command: os.path.join('./configs', opts.name + '.yaml')
        folders = [base / lit for lit in re.findall(r"""["']([\w./-]+)["']""", src)
                   for base in (cwd, script.parent) if (base / lit).is_dir()]
        values = [a.split("=", 1)[-1] for a in command[1:] if not a.startswith("-") or "=" in a]
        for folder in folders:
            for v in values:
                for ext in (".yaml", ".yml", ".json"):
                    if v and "/" not in v and (folder / f"{v}{ext}").is_file():
                        files.append(folder / f"{v}{ext}")
        hydra = re.search(r"hydra\.main\((.*?)\)", src, re.S)
        if hydra:
            path = re.search(r"config_path\s*=\s*[\"']([^\"']+)", hydra.group(1))
            name = re.search(r"config_name\s*=\s*[\"']([^\"']+)", hydra.group(1))
            over = [a.split("=", 1)[1] for a in command if a.startswith(("--config-name=", "-cn="))]
            if "-cn" in command:
                over.append(command[command.index("-cn") + 1])
            if "--config-name" in command:
                over.append(command[command.index("--config-name") + 1])
            if path:
                folder = (script.parent / path.group(1)).resolve()
                stem = (over or [name.group(1) if name else "config"])[0]
                for cand in (folder / f"{stem}.yaml", folder / f"{stem}.yml"):
                    if cand.is_file():
                        files.append(cand)
                        cfg = _load_config(cand) or {}
                        # Hydra defaults name further files in the same folder.
                        for item in cfg.get("defaults", []) if isinstance(cfg, dict) else []:
                            if isinstance(item, dict):
                                for group, choice in item.items():
                                    sub = folder / str(group).lstrip("/").split("@")[0] / f"{choice}.yaml"
                                    if sub.is_file():
                                        files.append(sub)
                            elif isinstance(item, str) and (folder / f"{item.lstrip('/')}.yaml").is_file():
                                files.append(folder / f"{item.lstrip('/')}.yaml")
    seen, out = set(), []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            out.append(f)
    return out


def _config_entries(cwd: Path, command: list[str]):
    """(file, key path, value) for every path-like value of the configuration,
    including command-line options such as ``--train_dir data/train`` and
    overrides such as ``data.root=/data`` (these first)."""
    args = list(command)
    for a in args:
        if not a.startswith("-") and "=" in a and "/" in a.split("=", 1)[1]:
            k, v = a.split("=", 1)
            yield None, k.lstrip("+~"), v
    for i, a in enumerate(args[:-1]):
        if a.startswith("--") and "=" not in a and "/" in args[i + 1]:
            yield None, "train." + a.lstrip("-"), args[i + 1]
    for a in args:
        if a.startswith("--") and "=" in a and "/" in a.split("=", 1)[1]:
            k, v = a.split("=", 1)
            yield None, "train." + k.lstrip("-"), v
    for f in _config_files(cwd, command):
        cfg = _load_config(f)
        if cfg is None:
            continue
        for key, value in _walk(cfg):
            if isinstance(value, str) and "/" in value and not value.startswith(("http", "//")):
                yield f, key, value


def _config_paths(cwd: Path, command: list[str]) -> list[str]:
    """Path-like strings in the configuration files the command names."""
    return [v for _, _, v in _config_entries(cwd, command)]


MODEL_FILES = ("*.safetensors", "*.bin", "*.ckpt", "*.pth", "model_index.json")
NOT_DATA = ("pretrain", "resume", "model", "log", "experiment", "checkpoint", "ckpt",
            "weight", "network", "save", "result", "output")


def _stream_from_path(p: Path) -> str | None:
    """A guess from folder names; words for a stored input are checked first,
    since "sharp_bicubic" names an input and "train_sharp" a target."""
    parts = [x.lower() for x in p.parts[-3:]]
    if any(w in x for x in parts for w in ("bicubic", "lq", "lr", "blur", "noisy",
                                            "degraded", "input", "x4", "x2")):
        return "lq"
    if any(w in x for x in parts for w in ("sharp", "gt", "hq", "hr", "clean",
                                            "target", "groundtruth")):
        return "gt"
    return None


def _clip_list(cwd: Path, command: list[str]) -> dict | None:
    """A clip list the run reads: "<clip> <frames> ..." per line in a text file
    named by the configuration, or a JSON index {clip: frames} beside the data."""
    for p in [q for q in cwd.glob("**/train.json") if q.is_file()][:5]:
        try:
            index = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(index, dict) and index and all(isinstance(v, int) for v in index.values()):
            names = list(index)[:8]
            return {"names": names, "frames": max(index[n] for n in names), "ext": ".jpg",
                    "template": "{:05d}", "from": str(p)}
    for s in _config_paths(cwd, command):
        # a CSV of videos, possibly named inside a list ("['a.csv', 'b.csv']")
        for c in re.findall(r"[\w./-]+\.csv", s):
            p = (cwd / c) if not c.startswith("/") else Path(c)
            if not p.is_file():
                continue
            import csv
            with open(p, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            videos = [v for row in rows[1:9] for v in row if Path(v).suffix.lower() in VIDEO_EXT]
            if videos:
                return {"videos": videos, "from": str(p)}
    for s in _config_paths(cwd, command):
        p = (cwd / s) if not s.startswith("/") else Path(s)
        if p.suffix.lower() == ".txt" and p.is_file():
            names, frames = [], []
            for line in p.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    names.append(parts[0]); frames.append(int(parts[1]))
            if names:
                return {"names": names, "frames": max(frames), "ext": ".png",
                        "template": "{:08d}", "from": str(p)}
    return None


def _resolved_entries(cwd: Path, command: list[str]) -> list[tuple]:
    """Configuration entries with simple ${key} references to other entries
    resolved; the command line comes first."""
    entries = list(_config_entries(cwd, command))
    values: dict = {}
    for _, k, v in entries:
        values.setdefault(k, v)
    resolve = lambda v: re.sub(r"\$\{([^}:]+)\}",                          # noqa: E731
                               lambda m: str(values.get(m.group(1), m.group(0))), v)
    return [(f, k, resolve(v)) for f, k, v in entries]


def discover_roots(cwd: Path, command: list[str], answers: Answers) -> dict[str, str]:
    """Data folders of the training set and the stream each holds."""
    entries = []
    for f, key, value in _resolved_entries(cwd, command):
        if "${" in value:
            continue
        low = key.lower()
        if Path(value).suffix.lower() in {".txt", ".json", ".yml", ".yaml", ".pth", ".pt",
                                          ".npy", ".pkl", ".ckpt", ".safetensors"}:
            continue
        if any(w in low for w in NOT_DATA) or value.lstrip().startswith("["):
            continue
        path = (cwd / value) if not value.startswith("/") else Path(value)
        if path.resolve() == cwd.resolve() or path.resolve() in cwd.resolve().parents:
            continue                                    # the project itself, or above it
        if path.is_dir() and any(next(path.glob(g), None) for g in MODEL_FILES):
            continue                                    # weights of a model
        entries.append((key, value))
    # The training set is what is probed; validation and test sets are not.
    train = [(k, v) for k, v in entries
             if "train" in k.lower() and not any(w in k.lower() for w in ("val", "test"))]
    chosen = train or entries
    candidates = []
    for key, value in chosen:
        path = (cwd / value).resolve() if not value.startswith("/") else Path(value)
        candidates.append((str(path), _stream_from_path(path), key))
    suggestion = {p: s for p, s, _ in candidates if s}
    sure = bool(train) and bool(suggestion) and len(suggestion) == len(candidates) \
        and "gt" in suggestion.values()
    if len(candidates) == 1 and not suggestion:
        folder = Path(candidates[0][0])
        subs = {str(d): _stream_from_path(d) for d in sorted(folder.iterdir())
                if d.is_dir()} if folder.is_dir() else {}
        subs = {d: st for d, st in subs.items() if st}
        if sorted(set(subs.values())) == ["gt", "lq"] and len(subs) == 2:
            # one folder whose subfolders are named for the target and the input
            suggestion, sure = subs, True
        else:
            # one folder, proposed as holding the target; the input is made from it
            suggestion = {str(folder): "gt"}
    return answers.get(
        "data_roots",
        "Which folders hold the training data, and which stream does each hold "
        "(\"gt\" for the target, \"lq\" for a stored input)? Found in the "
        f"configuration: {[(k, p) for p, _, k in candidates]}", suggestion or None, sure=sure)


def other_roots(cwd: Path, command: list[str], train: dict[str, str]) -> dict[str, str]:
    """Validation and test folders named by the configuration that do not exist
    here. Some runs build those sets before the training loader; they are never
    probed."""
    out = {}
    for f, key, value in _resolved_entries(cwd, command):
        low = key.lower()
        if not any(w in low for w in ("val", "test")) or any(w in low for w in NOT_DATA):
            continue
        if "${" in value or Path(value).suffix:
            continue
        path = (cwd / value).resolve() if not value.startswith("/") else Path(value)
        if _holds_data(path) or str(path) in train or path == cwd:
            continue
        out[str(path)] = _stream_from_path(path) or "gt"
    # a folder that holds another one named here is not a data folder itself
    return {p: st for p, st in out.items()
            if not any(q != p and q.startswith(p + os.sep) for q in out)}


def apply_overrides(cwd: Path, command: list[str], answers: Answers) -> list[str]:
    """The command with its configuration rewritten by the answers' ``set``.

    Data stored in an LMDB database cannot be served from a fixture, so a
    folder of images is proposed in its place when the configuration names one.
    """
    lmdb = {}
    for f, key, value in _config_entries(cwd, command):
        if value.endswith(".lmdb") and "train" in key.lower():
            lmdb[key] = value[:-5]
            parent = key.rsplit(".", 1)[0]
            lmdb[f"{parent}.io_backend.type"] = "disk"
    sets = dict(answers.data.get("set") or {})
    if lmdb:
        sets.update(answers.get("set_lmdb", "The training data are read from LMDB, which "
                                "the wizard cannot serve. Which configuration values "
                                "should change (key path: value) to read folders of "
                                "images instead?", lmdb) or {})
    if not sets:
        return command
    out = list(command)
    for i, arg in enumerate(command):
        f = cwd / arg
        if f.suffix.lower() not in CONFIG_EXT or not f.is_file():
            continue
        cfg = _load_config(f)
        for key, value in sets.items():
            node = cfg
            parts = key.split(".")
            for part in parts[:-1]:
                node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
            node[parts[-1]] = value
        copy = f.with_name(f".ddc_wizard_{f.name}")
        if f.suffix.lower() == ".json":
            copy.write_text(json.dumps(cfg, indent=2))
        else:
            import yaml
            copy.write_text(yaml.safe_dump(cfg, sort_keys=False))
        CREATED.append(copy)
        out[i] = str(copy.relative_to(cwd)) if not Path(arg).is_absolute() else str(copy)
    return out


def served_settings(cwd: Path, answers: Answers) -> dict[str, str]:
    """Configuration files as the run should read them, changed by the answers'
    ``set_in`` ({"<file>:<key path>": value}). They are served when the run
    reads them, so this works for any file the project opens: one named on the
    command line, one named in the script, or one Hydra composes. Nothing on
    disk changes."""
    changed: dict[Path, list] = {}
    for spec, value in (answers.data.get("set_in") or {}).items():
        name, _, key = spec.rpartition(":")
        f = (cwd / name) if not name.startswith("/") else Path(name)
        changed.setdefault(f.resolve(), []).append((key, value))
    out = {}
    for f, items in changed.items():
        # a configuration the wizard rewrote for this run is read from its copy
        copy = f.with_name(f".ddc_wizard_{f.name}")
        targets = [f, copy] if copy.exists() else [f]
        source = copy if copy.exists() else f
        if f.suffix == ".py":
            text = source.read_text()
            for key, value in items:
                literal = repr(value) if not isinstance(value, bool) else str(value)
                text = re.sub(rf"\b({re.escape(key.split('.')[-1])}\s*=\s*)"
                              r"(-?\d+(?:\.\d+)?|True|False|'[^']*'|\"[^\"]*\")",
                              lambda m: m.group(1) + literal, text)
            out.update({str(t): text for t in targets})
            continue
        cfg = _load_config(source)
        for key, value in items:
            node = cfg
            parts = key.split(".")
            for part in parts[:-1]:
                node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
            node[int(parts[-1]) if isinstance(node, list) else parts[-1]] = value
        if f.suffix.lower() == ".json":
            text = json.dumps(cfg, indent=2)
        else:
            import yaml
            # leading comments can carry directives (Hydra's "# @package")
            head = []
            for line in source.read_text().splitlines():
                if not line.startswith("#"):
                    break
                head.append(line + "\n")
            text = "".join(head) + yaml.safe_dump(cfg, sort_keys=False)
        out.update({str(t): text for t in targets})
    return out


def _holds_data(root: Path) -> bool:
    """Real data are never added to. A folder whose images are all empty files
    holds stand-ins from an earlier run."""
    return root.exists() and any(
        f.is_file() and f.suffix.lower() in IMAGE_EXT | VIDEO_EXT | {".zip"}
        and f.stat().st_size > 0
        for f in root.rglob("*"))


def standin_tree(roots: dict[str, str], answers: Answers,
                 suggestion: dict | None = None, sure: bool = True) -> list[dict]:
    """Create empty files for data folders that do not exist on this machine,
    and return the layouts used. A suggestion is used without asking when
    ``sure``; one read from a clip index is proposed instead, since the index
    does not say whether a clip is stored as a folder or an archive."""
    used = []
    for root, stream in roots.items():
        root = Path(root)
        if _holds_data(root):
            continue
        layout = answers.get(
            f"layout:{root}",
            f"{root} does not exist here. Its layout: "
            "{\"clips\": number of clip folders, \"frames\": frames per clip, "
            "\"ext\": \".png\", \"names\": optional list of clip names, "
            "\"template\": frame name with {:08d}}; or {\"images\": number} for a flat folder; "
            "or {\"videos\": [video files, relative to the folder]}",
            suggestion or {"clips": 4, "frames": 100, "ext": ".png", "template": "{:08d}"},
            sure=sure and suggestion is not None)
        top = root
        while not top.parent.exists():
            top = top.parent
        if root.exists():
            # an existing folder without images: only what is added is removed
            before = set(root.rglob("*"))
            STANDIN_IN_EXISTING.append((root, before))
        else:
            CREATED.append(top)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # a path of another machine, such as the authors': where the data
            # are on this one is the person's to say
            raise Question("set", f"{root} cannot be created on this machine. Which "
                           "configuration values (key path: value) should point to a folder "
                           "here instead?", None)
        used.append(layout)
        if "zips" in layout:
            for name in layout["zips"]:
                (root / f"{name}.zip").touch()
            continue
        if "videos" in layout:
            for rel in layout["videos"]:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).touch()
            continue
        if "images" in layout:
            for i in range(int(layout["images"])):
                (root / f"{i:08d}{layout.get('ext', '.png')}").touch()
            continue
        names = layout.get("names") or [f"{i:03d}" for i in range(int(layout["clips"]))]
        tmpl = layout.get("template", "{:08d}")
        for c in names:
            d = root / c
            if not d.exists():
                CREATED.append(d)
            d.mkdir(parents=True, exist_ok=True)
            for i in range(int(layout["frames"])):
                (d / (tmpl.format(i) + layout.get("ext", ".png"))).touch()
    return used


def standin_lists(cwd: Path, command: list[str], answers: Answers) -> dict[str, str]:
    """Image lists named by the training configuration that do not exist here.

    Each is written with paths to empty stand-in files in a folder of the
    wizard's own, which is then served as a data folder.
    """
    roots = {}
    for f, key, value in _config_entries(cwd, command):
        path = (cwd / value) if not value.startswith("/") else Path(value)
        if (path.suffix.lower() != ".txt" or "train" not in key.lower()
                or any(w in key.lower() for w in ("val", "test"))):
            continue
        folder = cwd / f".ddc_wizard_{path.stem}"
        if path.exists():
            # a list this wizard wrote in an earlier run is used again; any
            # other list is the project's own and is left alone
            listed = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            if listed and all(Path(n).parent == folder for n in listed):
                LISTS.append(str(path))
                spec = answers.data.get(f"list:{path}")
                roots[str(folder)] = spec.get("stream", "gt") if isinstance(spec, dict) else "gt"
            continue
        spec = answers.get(
            f"list:{path}",
            f"{path} does not exist here. If it lists image files, how many, and which "
            "stream (\"gt\" or \"lq\")? Answer {\"images\": n, \"stream\": \"gt\"} or null",
            {"images": 16, "stream": "gt"})
        if not spec:
            continue
        folder.mkdir(exist_ok=True)
        CREATED.append(folder)
        names = [folder / f"{i:08d}.png" for i in range(int(spec["images"]))]
        for n in names:
            n.touch()
        top = path.parent
        while not top.parent.exists():
            top = top.parent
        if not top.exists():
            CREATED.append(top)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent == top and top.exists() and top not in CREATED:
            CREATED.append(path)
        path.write_text("".join(f"{n}\n" for n in names))
        LISTS.append(str(path))
        roots[str(folder)] = spec.get("stream", "gt")
    return roots


def standin_files(cwd: Path, answers: Answers) -> dict[str, str]:
    """Small files the user describes (an index of clips, for instance), written
    where the project expects them when they do not exist. One that exists is
    left as it is, and its described content is what the run reads (returned
    by resolved path)."""
    served = {}
    for rel, content in (answers.data.get("files") or {}).items():
        path = (cwd / rel) if not rel.startswith("/") else Path(rel)
        text = content if isinstance(content, str) else json.dumps(content)
        if path.exists():
            served[str(path.resolve())] = text
            continue
        top = path
        while not top.parent.exists():
            top = top.parent
        CREATED.append(top)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return served


def _has_key(cfg, key: str) -> bool:
    node = cfg
    for part in key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return False
    return True


def connect(cwd: Path, command: list[str], answers: Answers) -> WizardAdapter:
    # ``set`` changes the configuration named on the command line; a file the
    # run finds otherwise (named in the script, or built from an option) gets
    # each key it has, supplied when it is read
    # The copy of the named configuration is written first, from the file on
    # disk; the changes of this run are then served over it, so that a copy
    # left by an earlier run is never read.
    SERVED_CONFIG.clear()
    command_named = list(command)
    command = apply_overrides(cwd, command, answers)
    named = {(cwd / a).resolve() for a in command_named}
    for f in _config_files(cwd, command_named):
        if f.resolve() in named:
            continue
        cfg = _load_config(f)
        for key, value in (answers.data.get("set") or {}).items():
            if cfg is not None and _has_key(cfg, key):
                answers.data.setdefault("set_in", {})[f"{f}:{key}"] = value
    SERVED_CONFIG.update(served_settings(cwd, answers))
    described = standin_files(cwd, answers)
    listed = standin_lists(cwd, command, answers)
    roots = dict(listed) if listed else discover_roots(cwd, command, answers)
    layouts = standin_tree(roots, answers, _clip_list(cwd, command), sure=False)
    others = other_roots(cwd, command, roots)
    if others and layouts:
        standin_tree(others, answers, layouts[0])
    data = VirtualData({**others, **roots})
    data.served.update(described)
    data.served.update(served_settings(cwd, answers))
    # A clip index with more clips than the stand-ins is read as listing only
    # those that have stand-ins; the file itself is left as it is.
    for layout in layouts:
        index = Path(layout.get("from", ""))
        if index.suffix == ".json" and layout.get("names"):
            full = json.loads(index.read_text())
            data.served[str(index.resolve())] = json.dumps(
                {n: full[n] for n in layout["names"] if n in full})
    # A blank frame lets the project build and read before any probe runs.
    blank = [np.full((64, 64, 3), 128, np.uint8)] * 2
    data.frames = {"gt": blank, "lq": blank}
    scale = 1
    if "lq" in roots.values():
        scale = int(answers.get(
            "lq_scale", "How many times smaller is the stored input than the target "
            "(1 unless the task is super-resolution)?", _config_scale(cwd, command),
            sure=True))
    data.min_size = int(answers.get(
        "min_frame_size", "The smallest picture side the loader accepts (the "
        "patch it crops), in target pixels", _config_patch(cwd, command) or None,
        sure=True) or 0)
    big = np.full((max(data.min_size, 64), max(data.min_size, 64), 3), 128, np.uint8)
    small = cv2.resize(big, (big.shape[1] // scale, big.shape[0] // scale))
    data.frames = {"gt": [big] * 2, "lq": [small] * 2}
    dataset = capture_dataset(cwd, command, data, answers.data.get("setup"), others)
    adapter = WizardAdapter(dataset, data, answers, name=type(dataset).__name__, scale=scale,
                            cwd=cwd)
    # Ask now which entries are the input and the target, not halfway
    # through a probe.
    with data.active():
        adapter.keys(dataset[0])
    return adapter


SIZE_KEYS = (r"gt_size|patch_size|gt_patch_size|crop_size|h_size|train_ps|img_size|"
             r"image_size|out_size|load_size|lq_patchsize|resolution")


def _config_texts(cwd: Path, command: list[str]) -> list[str]:
    """The configuration as text: its files, Python configuration files named on
    the command line (not the script), and the command itself."""
    texts = [p.read_text(errors="ignore") for p in _config_files(cwd, command)]
    texts += [(cwd / a).read_text(errors="ignore") for a in command[1:]
              if a.endswith(".py") and (cwd / a).is_file() and a != command[1]]
    texts.append(" ".join(command))
    return texts


def _config_patch(cwd: Path, command: list[str]) -> int:
    """The largest patch or output size the configuration or the command names."""
    sizes = []
    for text in _config_texts(cwd, command):
        for m in re.finditer(rf"""["']?\b(?:{SIZE_KEYS})["']?\s*[:=\s]\s*\[?\s*(\d+)""",
                             text, re.I):
            sizes.append(int(m.group(1)))
        # an output frame given as width and height, as video inpainting does
        w = re.search(r"""["']w["']\s*:\s*(\d+)""", text)
        h = re.search(r"""["']h["']\s*:\s*(\d+)""", text)
        if w and h:
            sizes.append(min(int(w.group(1)), int(h.group(1))))
    return max(sizes) if sizes else 0


def _config_scale(cwd: Path, command: list[str]) -> int:
    """The ``scale`` the configuration sets, if it sets one."""
    for text in _config_texts(cwd, command):
        m = re.search(r"""["']?\bscale\b["']?\s*[:=\s]\s*(\d+)""", text)
        if m:
            return int(m.group(1))
    return 1


def _snapshot(root: Path) -> set[Path]:
    """The entries of a project's first two levels."""
    out = set()
    for first in root.iterdir():
        out.add(first)
        if first.is_dir() and not first.is_symlink() and first.name != ".git":
            try:
                out.update(first.iterdir())
            except OSError:
                pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cwd", type=Path, required=True)
    ap.add_argument("--answers", type=Path, default=Path("ddc_wizard.json"))
    ap.add_argument("--profile", type=Path, help="write the loader profile here")
    ap.add_argument("--contract", type=Path,
                    help="requirements to add to, or replace, those the configuration states")
    ap.add_argument("--certificate", type=Path,
                    help="write a certificate: the profile, and a verdict for every requirement")
    ap.add_argument("--follow-config", action="store_true",
                    help="change each value the configuration states for the loader, and "
                         "check that the delivery follows it")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    answers = Answers(args.answers)
    cwd = args.cwd.resolve()
    project_before = _snapshot(cwd)
    try:
        adapter = connect(cwd, command, answers)
        from .profile import describe_loader
        profile = describe_loader(adapter)
    except Question as q:
        print(f"QUESTION [{q.key}]: {q.text}")
        if q.suggestion is not None:
            print(f"  detected: {json.dumps(q.suggestion)}")
        print(f"  add it to {args.answers} and run again")
        return 2
    finally:
        # Nothing is ever deleted. What this run created is listed, so that it
        # can be removed by hand: the wizard's own stand-ins and configuration
        # copies, and whatever the training command made as it started.
        made = sorted({str(p) for p in CREATED}
                      | {str(root) + " (stand-in files added)" for root, _ in STANDIN_IN_EXISTING}
                      | {str(p) for p in _snapshot(cwd) - project_before})
        if made:
            report = args.answers.with_suffix(".created.txt")
            report.write_text("".join(f"{m}\n" for m in made))
            print(f"created by this run (not removed; listed in {report}):\n  "
                  + "\n  ".join(made), file=sys.stderr)
    answers.save()
    if answers.detected:
        print("used without asking (see the answers file): " + ", ".join(answers.detected),
              file=sys.stderr)
    from .requirements import combine, from_configuration, judge, load_contract
    contract = load_contract(args.contract)
    if contract.get("vary"):
        profile["variations"] = vary(args, command, contract["vary"], profile)
    configured, conflicting = from_configuration(cwd, command)
    requirements = combine(configured, contract, str(args.contract or "contract"))
    results = judge(profile, requirements)
    if args.follow_config:
        from .checks import CheckResult
        # the loader's data folders, and the file lists it reads them from
        roots = [r for r, _ in adapter.data.roots] + LISTS
        followed, switched = follow_configuration(args, command, configured, profile, roots)
        for f in followed:
            results.append(CheckResult(f"follows:{f['requirement']}", f["status"], f["note"],
                                       {k: v for k, v in f.items() if k != "note"}))
        profile["switched_settings"] = switched
    text = json.dumps(profile, indent=2, default=str)
    if args.profile:
        args.profile.write_text(text + "\n")
    print(text)
    for r in results:
        print(f"{r.status:5s} {r.message}", file=sys.stderr)
    for name, values in conflicting.items():
        print(f"not required: the configuration states {name} more than once, "
              f"differently: {values}", file=sys.stderr)
    if args.certificate:
        from .certificate import Certificate
        cert = Certificate(adapter.name, "the requirements of the project's configuration"
                           + (f" and of {args.contract}" if args.contract else ""))
        for r in results:
            cert.add(r)
        cert.profile = profile
        cert.write(args.certificate)
    return 0


def _changed_run(args, command: list[str], setting: dict, value, expect: dict | None):
    """Run the wizard again with one setting changed; return its profile and,
    when ``expect`` names requirements, their verdicts."""
    import subprocess
    import tempfile

    answers = json.loads(args.answers.read_text()) if args.answers.exists() else {}
    cmd = list(command)
    kind, key = setting["kind"], setting["key"]
    if kind == "none":                                     # the run as it is
        pass
    elif kind == "option":                                   # --name value / --name=value
        hit = [i for i, a in enumerate(cmd) if a == key or a.startswith(key + "=")]
        if hit and "=" in cmd[hit[0]]:
            cmd[hit[0]] = f"{key}={value}"
        elif hit:
            cmd[hit[0] + 1] = str(value)
        else:
            cmd += [key, str(value)]
    elif kind == "override":                               # name=value (Hydra)
        cmd = [a for a in cmd if not a.startswith(key + "=")] + [f"{key}={value}"]
    elif kind == "file":                                   # "<file>:<key path>"
        answers.setdefault("set_in", {})[key] = value
    else:                                                  # key path of the named config
        answers.setdefault("set", {})[key] = value
    answers.update(setting.get("answers", {}).get(json.dumps(value), {}))
    with tempfile.TemporaryDirectory() as td:
        a_file, p_file, c_file = (Path(td) / n for n in ("a.json", "p.json", "c.json"))
        a_file.write_text(json.dumps(answers))
        extra = []
        if expect:
            k_file = Path(td) / "contract.json"
            k_file.write_text(json.dumps({"from_config": False, "requirements": expect}))
            extra = ["--contract", str(k_file), "--certificate", str(c_file)]
        run = subprocess.run([sys.executable, "-m", "treatment_identity.wizard", "--cwd",
                              str(args.cwd), "--answers", str(a_file), "--profile", str(p_file),
                              *extra, "--", *cmd], capture_output=True, text=True,
                             env=ENVIRONMENT)
        return {"profile": json.loads(p_file.read_text()) if p_file.exists() else None,
                "verdicts": [{"requirement": g["gate"].split(":", 1)[1], "status": g["status"],
                              "observed": g["evidence"].get("observed")}
                             for g in json.loads(c_file.read_text())["gates"]]
                if expect and c_file.exists() else None,
                "error": None if run.returncode == 0
                else (run.stderr.strip().splitlines() or ["?"])[-1][:300]}


def vary(args, command: list[str], variations: dict, profile: dict) -> list[dict]:
    """Run again with each value of each setting a contract lists under
    ``vary``, and report what in the delivery changes. ``--name`` is a
    command-line option, ``name=`` a Hydra override, ``<file>:<key>`` a key of
    any configuration file, and any other name a key of the configuration the
    command names."""
    from .requirements import compare_profiles

    out = []
    for key, values in variations.items():
        kind = ("option" if key.startswith("--") else "override" if key.endswith("=")
                else "file" if ":" in key else "set")
        setting = {"kind": kind, "key": key.rstrip("=")}
        runs = [_changed_run(args, command, setting, v, None) for v in values]
        comparisons = [{"from": values[0], "to": v, "error": r["error"],
                        "changes": compare_profiles(runs[0]["profile"], r["profile"])
                        if runs[0]["profile"] and r["profile"] else None}
                       for v, r in zip(values[1:], runs[1:])]
        note = (f"{key} changes the delivery" if any(c["changes"] for c in comparisons)
                else f"{key} changes nothing in the delivery"
                if all(c["changes"] == {} for c in comparisons)
                else f"{key}: not every value could be run")
        out.append({"setting": key, "values": values, "comparisons": comparisons, "note": note})
        print(note, file=sys.stderr)
    return out


def _setting_for(source: str, key: str, command: list[str]) -> dict:
    if source == "command line":
        hit = any(a == f"--{key}" or a.startswith(f"--{key}=") for a in command)
        return {"kind": "option", "key": f"--{key}"} if hit else {"kind": "override", "key": key}
    return {"kind": "file", "key": f"{source}:{key}"}


def follow_configuration(args, command: list[str], configured: dict, profile: dict,
                         roots: list[str]) -> tuple[list[dict], list[dict]]:
    """Change the configuration and run again, one setting at a time.

    For every requirement the configuration states, the delivery must follow
    the new value: the code reads the setting. Every other true/false setting
    of the training configuration is switched too, and what changes in the
    delivery is reported. A first run with nothing changed tells whether the
    delivery repeats under the fixed seed; only then does an unchanged
    delivery mean that a setting changes nothing.
    """
    from .requirements import ALTERNATIVE, compare_profiles, training_switches

    control = _changed_run(args, command, {"kind": "none", "key": ""}, None, None)
    repeats = bool(control["profile"]) and \
        compare_profiles(profile, control["profile"]).get("delivered values") is None
    followed = []
    for name, req in configured.items():
        candidates = ALTERNATIVE[name](req) if name in ALTERNATIVE else []
        if not candidates:
            continue
        source, _, key = req["source"].partition(": ")
        setting = _setting_for(source, key, command)
        for raw, declared in candidates:
            if name == "scale":
                setting["answers"] = {json.dumps(raw): {"lq_scale": raw}}
            result = _changed_run(args, command, setting, raw, {name: declared})
            if not result["error"]:
                break
        verdict = (result["verdicts"] or [{}])[0] if result["verdicts"] else {}
        follows = verdict.get("status") == "PASS"
        followed.append({
            "requirement": name, "setting": req["source"], "from": req["declared"],
            "to": raw, "expected": declared, "observed": verdict.get("observed"),
            "status": "PASS" if follows else "FAIL" if verdict else "SKIP",
            "error": result["error"],
            "note": (f"{req['source']} changed to {raw!r}: the delivery follows" if follows else
                     f"{req['source']} changed to {raw!r}: the delivery does not follow "
                     f"(observed {verdict.get('observed')!r})" if verdict else
                     f"{req['source']} changed to {raw!r}: the run failed: {result['error']}")})
        print(followed[-1]["note"], file=sys.stderr)
    known = {r["source"] for r in configured.values()}
    answers = json.loads(args.answers.read_text()) if args.answers.exists() else {}
    fixed = set(answers.get("set") or {}) | set(answers.get("set_lmdb") or {})
    switched = []
    switches = training_switches(args.cwd.resolve(), command, roots)
    print(f"{len(switches)} true/false setting(s) in the loader's part of the configuration "
          f"(data folders: {', '.join(roots) or 'none'})", file=sys.stderr)
    for sw in switches:
        label = f"{sw['source']}: {sw['key']}"
        if label in known or sw["key"] in fixed:
            continue                     # judged above, or fixed for this run
        if "shuffle" in sw["key"].lower():
            switched.append({"setting": label, "effect": "not switched",
                             "note": "the order in which the DataLoader draws items; DDC asks "
                                     "the dataset for items by index and does not see it"})
            continue
        result = _changed_run(args, command, _setting_for(sw["source"], sw["key"], command),
                              not sw["value"], None)
        changes = compare_profiles(profile, result["profile"]) if result["profile"] else None
        if changes and not repeats:
            # the delivered values differ between any two runs of this loader
            changes.pop("delivered values", None)
        built = "loaded its data when it was built"
        if result["error"]:
            effect = "run failed"
        elif result["profile"] and built in json.dumps(result["profile"]):
            effect = "not observable: with this value the dataset " + built
        elif changes:
            effect = "changes the delivery"
        else:
            effect = "changes nothing in the delivery" if repeats else \
                "changes nothing the profile measures"
        switched.append({"setting": label, "from": sw["value"], "to": not sw["value"],
                         "effect": effect, "changes": changes, "error": result["error"]})
        print(f"{label} switched to {not sw['value']}: {effect}", file=sys.stderr)
    return followed, [{"delivery repeats under the fixed seed": repeats}] + switched

if __name__ == "__main__":
    # Run the package's module, not this copy of it that ``python -m`` loads
    # as __main__, so that the modules importing it share its state.
    from treatment_identity.wizard import main as _main
    raise SystemExit(_main())
