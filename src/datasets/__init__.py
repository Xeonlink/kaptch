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

app = typer.Typer(help="데이터셋 관리 도구", rich_markup_mode="rich")
console = Console()


def _get_dataset_info(dataset: Path) -> tuple[str, str, str, str, str]:
    """데이터셋의 정보를 분석합니다.

    Parameters:
        dataset (Path): 데이터셋 디렉토리 경로

    Returns:
        tuple[str, str, str, str, str]: (총개수, 라벨됨, 라벨안됨, 너비, 높이)
    """
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


def _get_current_count(dataset_path: Path) -> int:
    """데이터셋의 현재 이미지 개수를 반환합니다.

    Parameters:
        dataset_path (Path): 데이터셋 디렉토리 경로

    Returns:
        int: 현재 데이터셋의 이미지 개수
    """
    with open(dataset_path / DATA_CSV, "r") as f:
        count = 0
        reader = csv.reader(f)
        for _ in reader:
            count += 1
    return count


@app.command("list", help="모든 데이터셋 목록을 표시합니다")
def list_datasets():
    """데이터셋 디렉토리에 있는 모든 데이터셋의 정보를 표시합니다.

    각 데이터셋에 대해 다음 정보를 보여줍니다:
    - 📁 이름: 데이터셋 폴더명
    - 📊 총 개수: 전체 이미지 개수
    - ✅ 라벨됨: 라벨이 있는 이미지 개수
    - ❌ 라벨안됨: 라벨이 없는 이미지 개수
    - 📏 너비: 이미지 너비 (다양한 경우 'various' 표시)
    - 📐 높이: 이미지 높이 (다양한 경우 'various' 표시)
    """
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


@app.command(help="모든 데이터셋 목록을 표시합니다 (list 명령어의 단축형)")
def ls():
    """list 명령어의 단축형입니다. 모든 데이터셋 목록을 표시합니다."""
    list_datasets()


@app.command(help="새로운 데이터셋을 생성합니다")
def create(name: str):
    """새로운 데이터셋을 생성합니다.

    Parameters:
        name (str): 생성할 데이터셋 이름 (하이픈은 언더스코어로 변환됨)

    Examples:
        python -m src.datasets create my-captcha-dataset
        python -m src.datasets create sci_evaluation

    생성된 데이터셋은 dataset/ 폴더 아래에 위치하며,
    data_list.csv 파일이 자동으로 생성됩니다.
    """
    name = name.replace("-", "_").strip()
    path = DATASET_ROOT / name
    path.mkdir(parents=True, exist_ok=True)

    if (path / DATA_CSV).exists():
        console.print(f"Dataset '{name}' already exists")
        console.print(f"Check existing datasets using `dataset show`")
        return

    (path / DATA_CSV).touch()  # create new file

    console.print(f"Dataset '{name}' created")
    console.print(f"Check existing datasets using `dataset show`")


@app.command(help="웹에서 캡챠 이미지를 자동으로 수집합니다")
def crawl(name: str, className: str | None = None, goal: int = 1100, headless: bool = False):
    """웹사이트에서 캡챠 이미지를 자동으로 수집합니다.

    Parameters:
        name (str): 데이터셋 이름
        className (str, optional): 사용할 POM 클래스명 (기본값: 데이터셋명)
        goal (int): 수집할 목표 이미지 개수 (기본값: 1100)
        headless (bool): 헤드리스 모드로 실행 (기본값: False)

    Examples:
        python -m src.datasets crawl sci
        python -m src.datasets crawl nice --goal 500 --headless
        python -m src.datasets crawl nhn_kcp --className NHN_KCP_Page

    지원하는 인증업체:
    - nhn_kcp: NHN KCP
    - nice: NICE 평가정보
    - sci: SCI 평가정보
    - kmcert: KMCERT

    수집된 이미지는 dataset/{name}/data/ 폴더에 저장되며,
    data_list.csv 파일에 경로와 빈 라벨이 추가됩니다.
    """
    path = DATASET_ROOT / name
    if not path.exists():
        console.print(f"Dataset '{name}' does not exist")
        return

    count = _get_current_count(path)
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


@app.command(help="웹 기반 라벨링 서버를 시작합니다")
def label(port: int = 3000, debug: bool = True):
    """웹 브라우저를 통해 캡챠 이미지에 라벨을 붙일 수 있는 서버를 시작합니다.

    Parameters:
        port (int): 서버 포트 번호 (기본값: 3000)
        debug (bool): 디버그 모드 활성화 (기본값: True)

    Examples:
        python -m src.datasets label
        python -m src.datasets label --port 8080 --debug False

    서버가 시작되면 웹 브라우저에서 http://localhost:{port}로 접속하여
    라벨링 작업을 진행할 수 있습니다.

    라벨링 완료 후에는 train 명령어로 모델 훈련을 시작할 수 있습니다.
    """
    start_server(port, debug)


@app.command(help="데이터셋을 삭제합니다")
def remove(name: str):
    """데이터셋과 관련된 모든 파일을 삭제합니다.

    Parameters:
        name (str): 삭제할 데이터셋 이름

    Examples:
        python -m src.datasets remove old-dataset
        python -m src.datasets rm test-dataset

    ⚠️  주의: 이 작업은 되돌릴 수 없습니다.
    데이터셋 폴더와 모든 이미지, 라벨 데이터가 영구적으로 삭제됩니다.
    """
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


@app.command(help="데이터셋을 삭제합니다 (remove 명령어의 단축형)")
def rm(name: str):
    """remove 명령어의 단축형입니다. 데이터셋을 삭제합니다."""
    remove(name)
