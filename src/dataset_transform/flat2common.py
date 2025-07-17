"""
dataset -> ** -> filename
filename format: {label}_{4자리_hash값}.png
filename format regex: r"^[0-9a-zA-Z]+_([A-Za-z0-9]{4}).png$"

위와 같은 형태의 dataset 폴더를 아래와 같은 형태의 dataset 폴더형태로 변환하는 코드

dataset/
├── train/
│   ├── 0000/
│   │   ├── 000.png
│   │   ├── 001.png
│   │   ├── ...
│   ├── 0001/
│   │   ├── 000.png
│   │   ├── 001.png
│   │   ├── ...
│   ├── ...
├── test/
│   ├── 0000/
│   │   ├── 000.png
│   │   ├── 001.png
│   │   ├── ...
│   ├── 0001/
│   │   ├── 000.png
│   │   ├── 001.png
│   │   ├── ...
│   ├── ...
├── train_list.csv
├── test_list.csv

train_list.csv, test_list.csv 는 header가 없는 2열 형식이다.
첫번째 열에는 dataset/ 에서 각 이미지까지의 상대경로가 적혀있고,
두번째 열에는 해당 이미지의 라벨이 적혀있다.
"""

import argparse
from typing import TypedDict, Callable
import os
import re
import shutil
import random
import csv
from typing import Tuple
import glob


class ParsedArgs(TypedDict):
    dataset: str
    output: str
    split: float


class Arguments:
    dataset: str
    output: str
    split_ratio: float

    def __init__(self, dataset: str, output: str, split: float):
        self._dataset = dataset
        self._output = output
        self._split_ratio = split

    @property
    def dataset_path(self) -> str:
        return self._dataset

    @property
    def output_path(self) -> str:
        if self._output is None:
            folder_name = os.path.basename(self._dataset)
            return os.path.join(self._dataset, "..", folder_name + "_converted")
        return self._output

    @property
    def split_ratio(self) -> float:
        return self._split_ratio

    @staticmethod
    def from_parsed(parsed_args: ParsedArgs) -> "Arguments":
        return Arguments(
            dataset=parsed_args.dataset,
            output=parsed_args.output,
            split=parsed_args.split,
        )


def copy_and_record(subset: list[Tuple[str, str, str]], subset_name: str, dataset_path: str, output_path: str):
    """
    subset의 이미지를 1000개씩 끊어서 0000, 0001, ... 폴더에 000.png ~ 999.png로 저장
    Parameters:
        subset (list[tuple[str, str, str]]): (relpath, label, hash) 리스트
        subset_name (str): 'train' 또는 'test'
        dataset_path (str): 원본 데이터셋 경로
        output_path (str): 출력 데이터셋 경로
    Returns:
        list[tuple[str, str]]: (상대경로, 라벨) 리스트
    """
    relpaths_labels: list[Tuple[str, str]] = []
    for idx, (relpath, label, _) in enumerate(subset):
        folder_idx = idx // 1000
        file_idx = idx % 1000
        folder_name = f"{folder_idx:03d}"
        file_name = f"{file_idx:03d}.png"
        folder_path = os.path.join(output_path, subset_name, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        src = os.path.join(dataset_path, relpath)
        dst = os.path.join(folder_path, file_name)
        shutil.copy2(src, dst)
        rel_output_path = os.path.relpath(dst, output_path)
        relpaths_labels.append((rel_output_path, label))

    return relpaths_labels


def write_csv(records: list[Tuple[str, str]], filename: str, output_path: str) -> None:
    """
    (상대경로, 라벨) 리스트를 csv 파일로 저장합니다.
    Parameters:
        records (list[tuple[str, str]]): (상대경로, 라벨) 리스트
        filename (str): 저장할 csv 파일명
        output_path (str): 출력 데이터셋 경로
    Returns:
        None
    """
    with open(os.path.join(output_path, filename), "w", newline="") as f:
        writer = csv.writer(f)
        for row in records:
            writer.writerow(row)


def main(args: Arguments) -> None:
    """
    데이터셋 폴더를 새로운 구조로 변환하고, train/test csv를 생성합니다.
    Parameters:
        args (Arguments): {'dataset': 원본 폴더, 'output': 출력 폴더}
    Returns:
        None
    """
    dataset_path = args.dataset_path
    output_path = args.output_path
    split_ratio = args.split_ratio

    # 1. 파일 목록 수집 및 라벨/해시 추출
    paths = glob.glob(os.path.join(dataset_path, "**", "*.png"), recursive=True)
    print(f"found {len(paths)} files")
    pattern = re.compile(r"^([0-9a-zA-Z]+)_([A-Za-z0-9]{4})\.png$")
    data: list[Tuple[str, str, str]] = []  # (relpath, label, hash)
    for path in paths:
        if not os.path.isfile(path):
            print(f"not file: skip {path}")
            continue
        relpath = os.path.relpath(path, dataset_path)
        filename = os.path.basename(path)
        if not pattern.match(filename):
            print(f"not match: skip {path}")
            continue
        label, hashval = pattern.match(filename).groups()
        data.append((relpath, label, hashval))

    # 2. train/test 분할
    random.shuffle(data)
    split_idx = round(len(data) * split_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    # 3. 폴더 생성 및 파일 복사
    train_records = copy_and_record(train_data, "train", dataset_path, output_path)
    test_records = copy_and_record(test_data, "test", dataset_path, output_path)

    # 5. CSV 파일 생성
    write_csv(train_records, "train_list.csv", output_path)
    write_csv(test_records, "test_list.csv", output_path)

    # 6. 결과 출력
    print(f"train: {len(train_records)}")
    print(f"test: {len(test_records)}")
    print(f"total: {len(train_records) + len(test_records)}")
    print(f"split ratio: {split_ratio}")
    print("Done!")


def range_float(min: float, max: float, include_min: bool = True, include_max: bool = True) -> Callable[[str], float]:
    def _range_float(x: str) -> float:
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Value {x} is not a float")

        if include_min and x < min:
            raise ValueError(f"Value {x} is less than {min}")
        elif not include_min and x <= min:
            raise ValueError(f"Value {x} is less than {min}")
        elif include_max and x > max:
            raise ValueError(f"Value {x} is greater than {max}")
        elif not include_max and x >= max:
            raise ValueError(f"Value {x} is greater than {max}")

        return x

    return _range_float


if __name__ == "__main__":
    split_arg_type = range_float(0.0, 1.0, False, False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="dataset directory")
    parser.add_argument("--output", type=str, required=False, default=None, help="output directory")
    parser.add_argument(
        "--split", type=split_arg_type, required=False, default=0.8, help="train/test split ratio (0.0 ~ 1.0)"
    )
    args: ParsedArgs = parser.parse_args()
    args = Arguments.from_parsed(args)
    main(args)
