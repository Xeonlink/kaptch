import typer
from rich.console import Console
from rich.tree import Tree
from pathlib import Path
from rich.style import Style
from src.constants import CHECKPOINT_ROOT
import shutil
from src.train.checkpoint import Checkpoint
from rich.table import Table
from rich import box
from typing_extensions import Annotated

console = Console()
app = typer.Typer(help="체크포인트 관리 도구", rich_markup_mode="rich")


def _tree_from_path(folder_path: Path, tree: Tree, style: Style) -> Tree:
    """폴더 경로로부터 트리 구조를 생성합니다.

    Parameters:
        folder_path (Path): 탐색할 폴더 경로
        tree (Tree): Rich Tree 객체
        style (Style): 적용할 스타일

    Returns:
        Tree: 업데이트된 트리 객체
    """
    for path in folder_path.iterdir():
        if path.is_dir():
            subtree = tree.add(path.name, style=style)
            _tree_from_path(path, subtree, style)
        else:
            tree.add(path.name, style=style)
    return tree


@app.command("list", help="체크포인트 목록을 표시합니다")
def list_checkpoints(
    name: Annotated[str, typer.Option(help="데이터셋 이름")] = "all",
):
    """특정 데이터셋의 모든 체크포인트 정보를 표시합니다.

    Parameters:
        name (str): 데이터셋 이름 또는 "all" (모든 데이터셋)

    Examples:
        python -m src.checkpoints list sci
        python -m src.checkpoints list all
        python -m src.checkpoints ls nice

    표시되는 정보:
    - Name: 체크포인트 파일명
    - Epoch: 훈련 에포크 수
    - Test Acc: 테스트 정확도
    - Avg Loss: 평균 손실값
    - LR: 학습률
    """

    if name == "all":
        dataset_paths = filter(lambda x: x.is_dir(), CHECKPOINT_ROOT.iterdir())
        for dataset_path in dataset_paths:
            list_checkpoints(dataset_path.name)

        console.print("[bold green]All checkpoints listed[/]")
        return

    table = Table(
        title=f"Checkpoints for [bold yellow]{name}[/]",
        title_style=Style(color="white", bold=False),
        header_style="green",
        box=box.ROUNDED,
    )
    table.add_column("Name")
    table.add_column("Epoch")
    table.add_column("Test Acc")
    table.add_column("Avg Loss")
    table.add_column("LR")

    checkpoints: list[Checkpoint] = []
    for checkpoint_path in (CHECKPOINT_ROOT / name).iterdir():
        checkpoint = Checkpoint.load(checkpoint_path)
        checkpoint.name = checkpoint_path.name
        checkpoints.append(checkpoint)

    prev_test_acc = 0
    for checkpoint in sorted(checkpoints, key=lambda x: x.epoch):
        if checkpoint.test_acc > prev_test_acc:
            test_acc_color = "green"
            prev_test_acc = checkpoint.test_acc
        elif checkpoint.test_acc == prev_test_acc:
            test_acc_color = "yellow"
        else:
            test_acc_color = "red"

        table.add_row(
            checkpoint.name,
            str(checkpoint.epoch),
            f"[{test_acc_color}]{checkpoint.test_acc:.4f}[/]",
            f"{checkpoint.avg_loss:.4f}",
            f"{checkpoint.lr:.2e}",
        )

    console.print(table)


@app.command(help="체크포인트 목록을 표시합니다 (list 명령어의 단축형)")
def ls(
    name: Annotated[str, typer.Option(help="데이터셋 이름")] = "all",
):
    """list 명령어의 단축형입니다. 체크포인트 목록을 표시합니다."""
    list_checkpoints(name)


@app.command(help="체크포인트를 삭제합니다")
def remove(name: str):
    """특정 데이터셋의 모든 체크포인트를 삭제합니다.

    Parameters:
        name (str): 삭제할 데이터셋 이름

    Examples:
        python -m src.checkpoints remove sci
        python -m src.checkpoints rm old-dataset

    ⚠️  주의: 이 작업은 되돌릴 수 없습니다.
    checkpoints/{name}/ 폴더와 모든 체크포인트 파일이 영구적으로 삭제됩니다.
    """
    if not typer.confirm(f"Are you sure you want to remove {name} checkpoints?"):
        console.print("[bold red]Remove cancelled[/]")
        return

    shutil.rmtree(CHECKPOINT_ROOT / name)
    console.print(f"[bold green]Checkpoints {name} removed[/]")


@app.command(help="체크포인트를 삭제합니다 (remove 명령어의 단축형)")
def rm(name: str):
    """remove 명령어의 단축형입니다. 체크포인트를 삭제합니다."""
    remove(name)


if __name__ == "__main__":
    app()
