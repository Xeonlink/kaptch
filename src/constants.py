from pathlib import Path

APP_NAME = "Kaptch"

DATA_ROOT = Path("data")

CHECKPOINT_ROOT = DATA_ROOT / "checkpoints"
DATASET_ROOT = DATA_ROOT / "dataset"
BLANK_INDEX = 10

DATA_CSV = "data_list.csv"

ENCODABLE_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CTC_BLANK_INDEX = len(ENCODABLE_CHARS)
