import os
import glob
import csv
import random
import string
import argparse
import shutil


def main(dataset_path: str) -> None:
    output_path = os.path.join(dataset_path, "..", os.path.basename(dataset_path) + "_flat")
    os.makedirs(output_path, exist_ok=True)

    csv_paths = glob.glob(os.path.join(dataset_path, "*.csv"))
    idx = 0
    for csv_path in csv_paths:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                relpath, label = row
                hashval = "".join(random.choices(string.ascii_letters + string.digits, k=4))
                folder_name = f"{idx // 1000:03}"
                file_name = f"{label}_{hashval}.png"
                folder_path = os.path.join(output_path, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                src = os.path.join(dataset_path, relpath)
                dst = os.path.join(folder_path, file_name)
                shutil.copy(src, dst)
                idx += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="dataset directory")
    args = parser.parse_args()

    main(args.dataset)
