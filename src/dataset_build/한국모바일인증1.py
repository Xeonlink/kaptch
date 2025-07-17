"""
한국모바일인증 데이터셋을 한국모바일인증1 데이터셋으로 변환하는 스크립트

데이터셋 구조:
- dataset/한국모바일인증1/{label}_XXXX.png
"""

import os
import csv
import shutil
import random
import string

ORIGINAL_DATASET_DIR: str = os.path.join("dataset", "한국모바일인증")
DATASET_DIR: str = os.path.join("dataset", "한국모바일인증1")


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(os.path.join(ORIGINAL_DATASET_DIR, "test_list.csv"), "r") as f:
        reader = csv.reader(f)
        for path, label in reader:
            h = "".join(random.choices(string.ascii_letters + string.digits, k=4))
            new_filename = f"{label}_{h}.png"
            src_path = os.path.join(ORIGINAL_DATASET_DIR, path)
            dst_path = os.path.join(DATASET_DIR, new_filename)
            shutil.copy(src_path, dst_path)


if __name__ == "__main__":
    main()
