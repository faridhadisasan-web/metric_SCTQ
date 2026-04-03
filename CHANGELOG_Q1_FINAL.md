# Q1 Final Changelog

## Major framework upgrades
- Added a three-video real article protocol (`run_real_article`) that discovers `video1/`, `video2/`, `video3/` and evaluates sparse, medium, and crowded scenes.
- Added frame sampling controls so real-video experiments can run on sampled subsets (default: 50 uniformly sampled frames per video).
- Added a separate clean real benchmark across videos and a multi-video corruption benchmark summary.
- Added high-level synthetic plots: ranking chart, component comparison, and SCTQ-vs-IDF1/IDP/IDR scatter plots.
- Reworked ablation reporting to generate concise outputs and an actual comparison figure.
- Reworked calibration reporting and added a report that explains why the selected weights are preferred over a pure Pearson-maximum choice.

## Runtime and HPC fixes
- Removed pandas and seaborn from the runtime path and requirements.
- Rewrote CSV aggregation/reporting using stdlib helpers.
- Added offline-safe YOLO loading with clear errors when local weights are missing.
- Reduced default real-video detector cost to `models/yolo11n.pt` at `640` resolution.

## Corruption-suite fixes
- Normalized corruption config shapes before execution.
- Added fail-fast validation when no noisy runs are produced.
- Precompute image-corrupted detections once per corruption/severity/run and reuse them across all trackers.
- Cache corrupted detections and clean detections separately.
- Added robustness summary outputs: mean drop, degradation slope, and area under the degradation curve for SCTQ and each component.

## Reporting and reproducibility
- Added clean output directory preparation for reproducible reruns.
- Added config/protocol snapshots to outputs.
- Added multi-video article report generation.
