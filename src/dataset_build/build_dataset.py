from playwright.sync_api import BrowserContext, Page, sync_playwright
from typing import Literal
from pom import PomFactory
import argparse
import csv
import os


class Arguments:
    dest: str
    authcom: PomFactory.Authcom
    train: int
    test: int
    headless: bool

    def __init__(self, args: argparse.Namespace):
        self.dest = args.dest
        self.authcom = args.authcom
        self.train = args.train
        self.test = args.test
        self.headless = args.headless


def current_count(dataset_path: str, train_or_test: Literal["train", "test"]) -> int:
    with open(os.path.join(dataset_path, f"{train_or_test}_list.csv"), "r") as f:
        lines = f.readlines()
    return len(lines)


def build_base_folder(dataset_path: str) -> None:
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "test"), exist_ok=True)

    if not os.path.exists(os.path.join(dataset_path, "train_list.csv")):
        open(os.path.join(dataset_path, "train_list.csv"), "w").close()
    if not os.path.exists(os.path.join(dataset_path, "test_list.csv")):
        open(os.path.join(dataset_path, "test_list.csv"), "w").close()


def main(context: BrowserContext, page: Page, args: Arguments) -> None:
    build_base_folder(args.dest)

    train_count = current_count(args.dest, "train")
    test_count = current_count(args.dest, "test")

    if train_count >= args.train and test_count >= args.test:
        print(f"Done!")
        print(f"train_count: {train_count}, test_count: {test_count}")
        print(f"train_goal_count: {args.train}, test_goal_count: {args.test}")
        return

    pom = PomFactory.create(args.authcom, context, page)
    pom.goto()
    pom.prepare()

    for subfolder in ["train", "test"]:
        with open(os.path.join(args.dest, f"{subfolder}_list.csv"), "a") as f:
            writer = csv.writer(f)
            for idx in range(train_count, args.train):
                folder_name = f"{idx // 1000:03d}"
                file_name = f"{idx % 1000:03d}.png"
                folder_path = os.path.join(args.dest, subfolder, folder_name)
                os.makedirs(folder_path, exist_ok=True)
                path = os.path.join(folder_path, file_name)
                path = os.path.relpath(path, args.dest)
                pom.save_captcha(path)
                writer.writerow([path, ""])

    browser.close()
    print(f"Done! train_count: {train_count}, test_count: {test_count}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, required=True, help="destination directory")
    parser.add_argument("--authcom", type=PomFactory.authcom_type, required=True, help="auth company name")
    parser.add_argument("--train", type=int, required=False, default=1_000, help="train dataset size")
    parser.add_argument("--test", type=int, required=False, default=100, help="test dataset size")
    parser.add_argument("--headless", type=bool, required=False, default=False, help="headless mode")
    args: Arguments = Arguments(parser.parse_args())

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context()
        page = context.new_page()
        main(context, page, args)
