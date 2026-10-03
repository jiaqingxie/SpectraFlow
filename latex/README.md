# Manuscript source snapshot

These files are the Overleaf source at commit
`b42f38bdc48aa32b04a250409a0ef2df00a43508`, copied without changing the manuscript's
scientific claims. All referenced figure PDFs, the bibliography, Springer Nature
class and bibliography styles are included. Authentication credentials and build
logs are excluded.

The [compiled snapshot PDF](manuscript_snapshot.pdf) was built with Tectonic 0.17.0;
its SHA256 and the source hashes are recorded in `build_manifest.json`.

The manuscript is an **archived draft, not a validated final submission**.
The audit found invalid identity pairing in the original 84-pair NIST result,
an ordering discrepancy in the principal preprocessing, incomplete archived
four-seed evidence, and an OOD test-protocol discrepancy. Read
[`../REPRODUCIBILITY_AUDIT.md`](../REPRODUCIBILITY_AUDIT.md) before using these
results. The corrected NIST subset does not establish unseen-molecule
generalization. The new audited figures are available separately in
[`../figures/reproduction/`](../figures/reproduction/); the original manuscript
figures are retained for traceability.

Build with a LaTeX installation that includes the required packages:

```bash
cd latex
pdflatex -interaction=nonstopmode -halt-on-error sn-article.tex
bibtex sn-article
pdflatex -interaction=nonstopmode -halt-on-error sn-article.tex
pdflatex -interaction=nonstopmode -halt-on-error sn-article.tex
```

Alternatively, from the repository root:

```bash
tectonic -X compile latex/sn-article.tex
```

`sn-article.pdf` is generated locally and ignored by Git. The source for the
overview figure is in `figs/figure1_tikz_source/`; compile that directory's
`Figure1_reference_tikz.tex` separately if changing the overview. Dataset and
figure provenance are described in the root README and audit.
