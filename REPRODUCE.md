# Minimal reproduction package

Partial Sensing-Plan Installation in Networked Estimation: Structural Loss and Compatibility Certificates

## Run

Use Python 3.9 and the pinned packages in requirements.txt (the tested environment).
From this extracted directory:

```text
python -m venv .venv
# Activate .venv according to your operating system.
python -m pip install -r requirements.txt
python reproduce.py
```

The runner needs only NumPy and Matplotlib. No ns-3 installation, network access,
absolute project path, sealed perception data, or GPU is used. A pre-existing
compatible environment can run `python reproduce.py` directly. Approximate runtime
is a few minutes on a desktop. Outputs go to results/ and figures/.

## Reproduced scope

- Exact rational witness, all 872 cycle permutations / 201,440 mask-parameter cases.
- 819 exhaustive small plan pairs, 600 seeded heterogeneous pairs and 31,374 masks;
  correlated expectations, component maxima, a nonproduct-reachability boundary,
  and the third-order parity witness.
- Original compatibility selection and fallible protocol checks, including cache/truth
  invariance and six staging branches; these are not new radio simulations.
- Figures 2-4, all 24 matched primary effects, all 18 mechanism-prevalence cells,
  all 1920 primary selection decisions and all eight processing-sensitivity pairs.
- Tables I-IV: rational witness, primary MSE/occupation, timing medians, sensitivity.
- Jensen and staged diagnostic summaries from the supplied frozen input tables.

The numerical runner verifies all generated CSV cells against expected/ to 1e-11
relative and 1e-12 absolute tolerance. It separately asserts the frozen integer
counts, 20 identical pairs and maximum nominal cost increase. Floating point
platform differences may affect the last digit of the structural residual.
Figure PDF bytes depend on font/Matplotlib versions and timestamps; numerical
content rather than identical PDF hashes is the reproducibility criterion.

## Data and limitations

Inputs retain unfavorable outcomes: 720 prior run summaries; all 108 targeted run
summaries (96 primary, four staged, eight sensitivity); 144,000 sampled prior action
ticks; 1920 selection rows; timing, Jensen, filter checks and staging events.
An action tick contains the three-source action vector; it is not an independent
process replicate. Four process instances, each with two nested radio seeds,
underlie the 24 primary host/compatibility pairs.

This compact package reconstructs results from frozen network outputs. It does
not rerun the original packet-level NR-V2X campaign, reconstruct all byte-level MAC
audits from packet traces, or establish hardware/scalability/safety performance.
The full simulator environment and larger packet/filter trace archive remain in
the research project and are intentionally excluded. The existing result mapping
records some full-archive provenance paths; those are descriptive, not required
inputs to this runner. `documentation/portable_result_mapping.csv` lists packaged
inputs. No proprietary perception datasets are included.

## Files and integrity

`manifest_sha256.csv` covers all supplied code/data/documents except itself and
generated output directories. `reproduce.py` verifies it before executing scripts.
`expected/` contains reference numeric outputs, not additional experiments.
Original controller modules retain their package-relative imports. The runner
disables bytecode writing. No software or data reuse license has been assigned; publication does not
imply a permission grant beyond applicable law. Existing notices are retained.

## Author-supplied framework (final submission update)

Figure 1 is supplied as `fig1.png`, byte-identical to the author's project-root
image, with `fig1_framework.tex` providing the scientific notation and return-flow
corrections as native LaTeX. It is conceptual artwork, not a numerical output.
The manuscript embeds these two files using graphicx and TikZ. The numerical
runner rebuilds Figures 2-4; it does not replace the author's Figure 1.

## Repository and downloadable package

Repository: [lanston12/tcns-sensing-plan-repro](https://github.com/lanston12/tcns-sensing-plan-repro).

The repository root contains the extracted minimal package. Run `python reproduce.py`
from this directory after installing `requirements.txt`. The bundled
`TCNS_minimal_reproducibility.zip` contains the same manifested inputs and can also
be downloaded and extracted to run the checks. Git metadata and generated outputs
are not part of that ZIP. The initial public version retains all 720 prior and 108
targeted frozen run summaries and all unfavorable matched effects.
