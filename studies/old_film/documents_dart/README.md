# Documents inspected for DART (Study 4)

The publication and the two README files at the audited commits, searched for
any statement that the training target is transformed (`sharpen*`,
`unsharp*`, `USM`) and for how inputs and targets are produced.

- `dart_arxiv_2607.21219.txt`: text of arXiv:2607.21219
  (https://arxiv.org/abs/2607.21219), extracted with `mutool draw -F txt`; the
  PDF itself is not redistributed, its SHA-256 is in `SHA256SUMS`.
- `dart_README.md`: DART at commit e17b95cee2c2.
- `absolutedegradation_README.md`: AbsoluteDegradation v1.0.0 (da0ad0a59585).

Result: "sharpen" appears only for a perceptual loss and for other methods'
outputs; no document states that the target is sharpened. The
AbsoluteDegradation README states that "Inputs and targets are grayscale".
