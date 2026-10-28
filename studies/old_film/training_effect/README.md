# Effect of two mismatches on trained models (appendix of the article)

Measurements made outside DDC, in the authors' earlier study of the old-film
restoration systems of Study 4, and reported in the article's appendix to show
why mismatches of the kinds DDC reports can matter. Neither is part of the
evaluation of DDC, and neither was rerun with DDC 1.11.0: they train and score
restoration models on a GPU.

- `e19_prefix_vs_window.py`, `e19_resampler_sensitivity.py`,
  `e19_resampler_sensitivity.csv`: RTN trained twice under each of three seeds,
  with its released fixed frame window and with the window its code draws;
  paired differences on twenty test clips, under the released scoring and at
  the references' own geometry.
- `geometry_fragility_v2_cells.csv`, `fragility_derive.py`,
  `validate_geometry.py`, `validated_counts.json`: 27 models (RTN, RRTN and
  MambaOFR architectures, each trained under the degradation of each system,
  three seeds) scored at 640x360 and at 640x368; counts of pairwise
  comparisons whose winner is reversed.

The trained models and their outputs are not included.
