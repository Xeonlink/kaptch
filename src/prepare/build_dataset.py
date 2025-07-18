from playwright.sync_api import BrowserContext, Page, sync_playwright
from pom import PomFactory
import argparse
import csv
import os


class Arguments:
    dest: str
    authcom: PomFactory.Authcom
    size: int
    headless: bool

    def __init__(self, args: argparse.Namespace):
        self.dest = args.dest
        self.authcom = args.authcom
        self.size = args.size
        self.headless = args.headless


def current_count(dataset_path: str) -> int:
    with open(os.path.join(dataset_path, "data_list.csv"), "r") as f:
        lines = f.readlines()
    return len(lines)


def build_base_folder(dest_path: str) -> None:
    os.makedirs(os.path.join(dest_path, "data"), exist_ok=True)

    if not os.path.exists(os.path.join(dest_path, "data_list.csv")):
        open(os.path.join(dest_path, "data_list.csv"), "w").close()


def main(context: BrowserContext, page: Page, args: Arguments) -> None:
    build_base_folder(args.dest)

    count = current_count(args.dest)
    if count >= args.size:
        print(f"Done!")
        print(f"count: {count}")
        print(f"goal_count: {args.size}")
        return

    pom = PomFactory.create(args.authcom, context, page)
    pom.goto()
    pom.prepare()

    with open(os.path.join(args.dest, "data_list.csv"), "a") as f:
        writer = csv.writer(f)
        for idx in range(count, args.size):
            print(f"\r데이터 수: {idx + 1:,}/{args.size:,}", end="", flush=True)
            folder_path = os.path.join(args.dest, "data", f"{idx // 1000:03d}")
            os.makedirs(folder_path, exist_ok=True)
            path = os.path.join(folder_path, f"{idx % 1000:03d}.png")
            pom.save_captcha(path)
            writer.writerow([os.path.relpath(path, args.dest), ""])

    browser.close()
    print(f"\r데이터 수: {args.size:,}/{args.size:,}", flush=True)
    print(f"Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, required=True, help="destination directory")
    parser.add_argument("--authcom", type=PomFactory.authcom_type, required=True, help="auth company name")
    parser.add_argument("--size", type=int, required=False, default=1_000, help="dataset size")
    parser.add_argument("--headless", action="store_true", help="headless mode")
    args: Arguments = Arguments(parser.parse_args())

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context()
        page = context.new_page()
        main(context, page, args)
