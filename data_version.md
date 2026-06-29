# data_version.md — CICIoMT2024 (canonical record)

**Date:** 2026-06-29
**Source:** Kaggle `limamateus/cic-iomt-2024-wifi-mqtt`, parquet (train+test).
**Nature:** CIC's official file-level 80/20 WiFi/MQTT split, concatenated across attacks.
train parquet = official 80%, test parquet = official 20%. 51 classes in `label`.
**Rejected:** `amineipad/...` (same merged data, CSV only); CIC direct site = registration form.
**Per-attack files:** not used (only behind CIC form); label column supplies attack class.

## Split-protocol feasibility (settled)
- Official CIC split (train vs test parquet; literature baseline): YES
- Random re-split (re-pool then split): YES
- Leakage-remediated split: YES
- Attack-family-held-out (headline shift axis): YES (via label families)
- Device-held-out: PCAP-gated -> DEFERRED
- Temporal: PCAP-gated -> DEFERRED
- Cross-protocol w/ Bluetooth: no Bluetooth CSVs -> DEFERRED

## Notes
- Execution-level attack-batch identity unavailable (merged); type-level via label only.
- Rows window-averaged (window 10 or 100 by attack) -> matters for near-duplicate audit.
- Frozen-dataset hashing at end of Phase 1 after Category-A remediation.


---
## Canonical freeze (2026-06-29) — notebook 05
- Category-A duplicates: ~36.9% train / 44.7% test exact feature-duplicates (feature-space collisions, uniform flood traffic). Official split cross-leak 461 rows (0.05%).
- **Frozen canonical = within-split deduped**: train 4,515,078 rows (md5 e49d08995a4efcef180a0d72b8632e3a), test 892,266 rows (md5 363df790fc8297e1f06958f600057111), in data/processed/. As-shipped parquet retained for the with-dups inflation comparison (C1 reports both).
- Cross-label feature collisions reported (accuracy ceiling).
- Single-feature probe = candidate discriminativeness ranking, NOT a leakage verdict; IAT is the prime artifact suspect. Legitimate-vs-shortcut decided by the C3 transfer/ablation test, not the probe.
- Feature dedup is keep-first on the 45 feature columns; labels/meta preserved.
