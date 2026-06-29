# data_version.md — CICIoMT2024 (canonical record)

**Date:** 2026-06-29
**Source:** Kaggle `limamateus/cic-iomt-2024-wifi-mqtt` (per-attack WiFi/MQTT CSVs, cited
across the published CICIoMT2024 literature; mirrors CIC's official csv/ release).
**Files:** 2 per-attack CSVs grouped by CIC's official file-level 80/20 train/test split.
**Rejected:** `amineipad/cic-iomt-dataset-2024` (2 merged feature-only files, random split,
no per-attack structure). CIC direct site is a registration form (no open directory).

## Split-protocol feasibility (settled at download)
- Random re-split: YES
- Official CIC train/test split (file-level 80/20; the literature baseline to audit): YES
- Leakage-remediated split: YES
- Attack-family-held-out (headline shift axis, replaces device-held-out): YES
- Device-held-out: PCAP-gated (no MAC) -> DEFERRED
- Temporal: PCAP-gated (no timestamp) -> DEFERRED
- Cross-protocol w/ Bluetooth: Bluetooth has no CSVs, diff schema -> DEFERRED

## Notes
- Labels encoded by filename (per-attack CSVs).
- Rows are window-averaged (window 10 or 100 by attack type) -> matters for near-duplicate audit.
- Canonical frozen-dataset hashing happens end of Phase 1 after Category-A remediation.
