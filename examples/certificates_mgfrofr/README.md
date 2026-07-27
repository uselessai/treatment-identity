# Superseded partial MgfrOFR probe

This directory preserves the first, partial MgfrOFR probe for historical
traceability. It predates the common eight-gate certificate and used an
incorrect declared target-transform count.

The canonical released-code certificate is now:

`../certificates_released/mgfrofr.json`

It was generated from the clean working tree at commit
`0302b90142d6bee69a6091790db621bfc031bf23`, treats the unclaimed precomputed
branch as `N/A`, and applies the same target, order, separability, temporal and
geometry contracts as the other released certificates. Do not use the partial
JSON in this directory for the manuscript table.
