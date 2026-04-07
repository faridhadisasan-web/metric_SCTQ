# Article trajectory visualization

This branch exports per-point raw track histories and adds a CLI to create paper-ready trajectory figures.

## 1) Run the clean article protocol

```bash
python -m sctq.cli.run_real_article \
  --config configs/default.yaml \
  --dataset-root . \
  --clips-per-video 3 \
  --max-frames 400 \
  --skip-corruptions
```

This produces per-clip folders such as:

```text
data/outputs/real_article/per_video/clean_sparse_video1_clip0/
```

Inside each run folder you will now also get:

```text
raw_tracks/*.csv
```

Each CSV stores all saved points of one tracker output:

- `track_id`
- `frame_idx`
- `cx`, `cy`
- `w`, `h`
- `conf`
- `class_id`

## 2) Export trajectory figures for the paper

### Batch mode (all clips and trackers)

```bash
python -m sctq.cli.export_trajectory_paths \
  --results-root data/outputs/real_article \
  --dataset-root . \
  --mode frame \
  --top-k 15 \
  --min-track-length 5
```

This creates PNG figures in:

```text
data/outputs/real_article/trajectory_plots/
```

### Single run / single tracker mode

```bash
python -m sctq.cli.export_trajectory_paths \
  --raw-tracks data/outputs/real_article/per_video/clean_sparse_video1_clip0/raw_tracks/clip0_sort_raw_tracks.csv \
  --video video1/video1.mp4 \
  --mode frame \
  --top-k 20 \
  --min-track-length 5 \
  --label-tracks
```

## 3) Suggested paper usage

For showing why persistence matters, compare the same clip across trackers:

- `sort`
- `centroid`
- `centroidkf`
- `bytetrack`
- `botsort`

Trackers with stronger persistence usually show longer, more continuous paths, while weaker trackers break a person into several short fragments.

## 4) Recommended figures

- one sparse clip
- one medium-density clip
- one crowded clip
- same clip across 2–4 trackers
- optionally use `--mode blank` if you want paths on a clean white background
