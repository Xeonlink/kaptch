import os
import csv
import typer
import importlib
import shutil
from pathlib import Path
from typing import Type
from rich.console import Console
from rich.table import Table, box
from rich.progress import track
from src.datasets.pom import Pom
from playwright.sync_api import sync_playwright
from PIL import Image
from src.datasets.label import start_server
from src.constants import DATASET_ROOT, DATA_CSV

app = typer.Typer()
console = Console()


def _get_dataset_info(dataset: Path) -> tuple[str, str, str, str, str]:
    labeled = 0
    unlabeled = 0
    sizes: set[tuple[int, int]] = set()

    with open(dataset / DATA_CSV, "r") as f:
        reader = csv.reader(f)
        for path, label in reader:
            image = Image.open(dataset / path)
            sizes.add((image.width, image.height))

            if label == "":
                unlabeled += 1
            else:
                labeled += 1

    if len(sizes) > 1:
        width, height = "various", "various"
    elif len(sizes) == 1:
        width, height = tuple(map(str, sizes.pop()))
    else:
        width, height = "None", "None"

    return str(labeled + unlabeled), str(labeled), str(unlabeled), str(width), str(height)


@app.command("list", help="List all datasets")
def list_datasets():
    datasets = [dataset for dataset in DATASET_ROOT.iterdir() if dataset.is_dir()]
    datasets = sorted(datasets, key=lambda x: x.name)

    table = Table(header_style="green", box=box.ROUNDED)
    table.add_column("📁 Name")
    table.add_column("📊 Total")
    table.add_column("✅ Labeled")
    table.add_column("❌ Unlabeled")
    table.add_column("📏 Width")
    table.add_column("📐 Height")

    for dataset in datasets:
        table.add_row(dataset.name, *_get_dataset_info(dataset))

    console.print(table)


@app.command(help="List all datasets (shortcut)")
def ls():
    list_datasets()


@app.command(help="Create a new dataset")
def create(name: str):
    name = name.replace("-", "_").strip()
    path = DATASET_ROOT / name
    path.mkdir(parents=True, exist_ok=True)

    if (path / DATA_CSV).exists():
        console.print(f"Dataset '{name}' already exists")
        console.print(f"Check existing datasets using `dataset show`")
        return

    open(path / DATA_CSV, "w").close()

    console.print(f"Dataset '{name}' created")
    console.print(f"Check existing datasets using `dataset show`")


@app.command(help="Crawl a dataset from web using playwright")
def crawl(name: str, className: str | None = None, goal: int = 1100, headless: bool = False):
    path = DATASET_ROOT / name
    if not path.exists():
        console.print(f"Dataset '{name}' does not exist")
        return

    def current_count(dataset_path: str) -> int:
        with open(os.path.join(dataset_path, DATA_CSV), "r") as f:
            count = 0
            reader = csv.reader(f)
            for _ in reader:
                count += 1
        return count

    count = current_count(path)
    if count >= goal:
        console.print(f"Done!")
        console.print(f"count: {count}")
        console.print(f"goal_count: {goal}")
        return

    try:
        module = importlib.import_module("pom", __file__)
        className = className or name.replace("-", "_")
        class_: Type[Pom] = getattr(module, className)
    except ImportError as e:
        console.print(e)
        console.print(f"Error importing module '{module}'")
        return
    except AttributeError as e:
        console.print(e)
        console.print(f"Error importing class '{className}'")
        return
    except Exception as e:
        console.print(e)
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()

        pom = class_(context, page)
        pom.goto()
        pom.prepare()

        with open(path / DATA_CSV, "a") as f:
            writer = csv.writer(f)
            for idx in track(range(count, goal), description="Crawling..."):
                folder_path = path / "data" / f"{idx // 1000:03d}"
                os.makedirs(folder_path, exist_ok=True)
                image_path = folder_path / f"{idx % 1000:03d}.png"
                pom.save_captcha(image_path)
                writer.writerow([image_path.relative_to(path), ""])

        browser.close()
        console.print(f"Done!")


@app.command(help="Start a labeling web server")
def label(port: int = 3000, debug: bool = True):
    start_server(port, debug)


@app.command(help="Remove a dataset")
def remove(name: str):
    path = DATASET_ROOT / name
    if not path.exists():
        console.print(f"Dataset '{name}' does not exist")
        return

    question = typer.confirm(f"Are you sure you want to delete dataset '{name}'?")
    if not question:
        return

    shutil.rmtree(path)

    console.print(f"Dataset '{name}' deleted")
    console.print(f"Check existing datasets using `dataset show`")


@app.command(help="Remove a dataset (shortcut)")
def rm(name: str):
    remove(name)
