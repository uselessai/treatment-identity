# Field certificates for treatment-identity 1.6.0

Eight schema-1.2 certificates record four released repositories and four
working trees. Each contains seven delivery-gate families (including three
pairwise separability records) plus geometry: ten gate records per document.
MgfrOFR is present in both scopes.

The content contracts assert normalised `[-1, 1]` values and the intended
luminance representation. Distinct RGB channels are not declared for this
old-film treatment. Separability uses 16 matched seeds and 4095 paired
randomisations.

Verify from the repository root:

```bash
python3 examples/certificates_1.6.0/verify_certificates.py
```
