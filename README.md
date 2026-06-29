# IoMT Compression Trustworthiness

Trustworthiness of compressed Internet-of-Medical-Things intrusion detectors:
what making a detector deployable (quantization, pruning, distillation) costs in
calibration, explanation fidelity, and safe abstention, and *why* the cost
concentrates where it does.

X-IDS Extension Paper 2. Primary dataset: CICIoMT2024. See the research plan for
the full design (research questions, contributions, methodology, phased plan).

## Layout

```
config/      paths.yaml (canonical paths) + datasets.yaml (sources/roles)
src/         iomtc_config.py (paths + constants), iomtc_metrics.py (calibration, CIs)
notebooks/   numbered, self-contained Colab stages
reports/     result + audit CSVs
figures/     plots
data/        datasets (gitignored, on Drive)
```

Code, CSVs, and figures are tracked in git. Datasets and model binaries live on
Google Drive and are gitignored. Every path resolves from `config/paths.yaml`;
nothing is hardcoded in notebooks.

## Notebooks

- `00_setup.ipynb` — one-time, self-contained: builds the folder skeleton, writes
  all `config/` and `src/` files (base64-embedded), sets git identity, stores the
  PAT to Drive, inits the repo + remote, first commit/push, self-test.
- `01_download_data.ipynb` — downloads CICIoMT2024, then runs the **inventory +
  metadata feasibility check** that determines whether device-held-out and
  temporal split protocols are possible from the CSVs.

## Reproducibility

Canonical seed 42; bootstrap B=1000; Option D calibration (Platt for n_calib<30,
isotonic otherwise). Constants are imported from `iomtc_config`, never re-typed.

License: MIT.
