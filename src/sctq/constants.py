import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Default config paths
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"
METRICS_CONFIG_PATH = CONFIGS_DIR / "metrics.yaml"
TRACKERS_CONFIG_PATH = CONFIGS_DIR / "trackers.yaml"
CORRUPTIONS_CONFIG_PATH = CONFIGS_DIR / "corruptions.yaml"
SYNTHETIC_CONFIG_PATH = CONFIGS_DIR / "synthetic.yaml"
REAL_VIDEO_CONFIG_PATH = CONFIGS_DIR / "real_video.yaml"

# Supported Trackers
MOTRACKERS_TRACKERS = ["sort", "iou", "centroid", "centroidkf"]
ULTRALYTICS_TRACKERS = ["bytetrack", "botsort"]
SUPPORTED_TRACKERS = MOTRACKERS_TRACKERS + ULTRALYTICS_TRACKERS

# SCTQ Default Constants (Can be overridden by metrics.yaml)
DEFAULT_ALPHA_P = 0.20
DEFAULT_BETA_1 = 0.5
DEFAULT_BETA_2 = 0.5
DEFAULT_TAU_PHI = 0.35
DEFAULT_TAU_A = 10.0  # Pixels per frame squared (arbitrary default, should be tuned)
DEFAULT_TAU_SHORT = 0.10
DEFAULT_TAU_SIZE = 10.0  # Pixels
DEFAULT_TAU_CONF = 0.2
DEFAULT_GAMMA_1 = 0.5
DEFAULT_GAMMA_2 = 0.5

# Level B Fragmentation Defaults
DEFAULT_TAU_T = 5
DEFAULT_TAU_S = 50.0
DEFAULT_TAU_THETA = 0.5
DEFAULT_W_T = 0.3
DEFAULT_W_S = 0.4
DEFAULT_W_THETA = 0.3

# SCTQ Weights
DEFAULT_W_P = 0.35
DEFAULT_W_D = 0.25
DEFAULT_W_F = 0.20
DEFAULT_W_C = 0.20
DEFAULT_LAMBDA_STAB = 0.20
