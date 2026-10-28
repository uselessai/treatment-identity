# Certificates regenerated under 1.5.0

These eight certificates were regenerated on 2026-08-11 in the clean locked
environment after the target-presence correction and schema 1.2 change.  The
generating tree was the candidate that identified itself as `1.5.0rc1` and was
tagged `v1.5.0` without further change; see
`environment/VALIDATION_2026-08-11.md`.  Certificates do not record a package
version, so no document here is affected by the rename.  The
historical schema 1.1 certificates remain in `examples/certificates_released`
and `examples/certificates_working`; they are not overwritten or relabelled.

- `released/` audits the four clean public commits.
- `working/` audits the four study working trees (MgfrOFR uses the same clean
  tree in both modes, as recorded by its certificate).

Every certificate validates against `treatment-certificate/1.2`, including the
now-required `undeclared_gates` field.  `verify_certificates.py` verifies the
eight hashes, schema-validates each document and checks that its ordered
`(gate, status)` outcomes match the corresponding historical certificate.

The audit command exits with status 1 by design when any certificate records a
divergence; that status is the reported result, not a failed regeneration.

```bash
python examples/certificates_1.5.0/verify_certificates.py
```
