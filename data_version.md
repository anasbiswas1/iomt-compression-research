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
