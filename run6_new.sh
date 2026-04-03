set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

module load python/3.10.20

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

export PYTHONUNBUFFERED=1

echo "Log file: $LOG_FILE"
echo "Started at: $(date)"
echo "PWD=$(pwd)"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"

echo "Python=$(which python)"
python --version

echo "=== Running tests ==="
pytest -q

echo "=== Running synthetic ==="
python -m sctq.cli.run_synthetic --config "$SCRIPT_DIR/configs/default.yaml"

echo "=== Running calibration ==="
python -m sctq.cli.run_calibration --config "$SCRIPT_DIR/configs/default.yaml"

echo "=== Running ablation ==="
python -m sctq.cli.run_ablation --config "$SCRIPT_DIR/configs/default.yaml"

echo "=== Clean-only three-video sanity run ==="
python -m sctq.cli.run_real_article \
  --config "$SCRIPT_DIR/configs/default.yaml" \
  --dataset-root "$SCRIPT_DIR" \
  --max-frames 500 \
  --skip-corruptions

echo "=== Final three-video article protocol ==="
python -m sctq.cli.run_real_article \
  --config "$SCRIPT_DIR/configs/default.yaml" \
  --dataset-root "$SCRIPT_DIR" \
  --max-frames 500

echo "Finished at: $(date)"