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

console = Console()
app = typer.Typer()


def _tree_from_path(folder_path: Path, tree: Tree, style: Style) -> Tree:
    for path in folder_path.iterdir():
        if path.is_dir():
            subtree = tree.add(path.name, style=style)
            _tree_from_path(path, subtree, style)
        else:
            tree.add(path.name, style=style)
    return tree


@app.command("list", help="List all checkpoints")
def list_checkpoints(name: str):

    if name == "all":
        for dataset_path in CHECKPOINT_ROOT.iterdir():
            if dataset_path.is_dir():
                list_checkpoints(dataset_path.name)

        console.print("[bold green]All checkpoints listed[/bold green]")
        return

    table = Table(
        title=f"Checkpoints for [bold yellow]{name}[/bold yellow]",
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

    for checkpoint in sorted(checkpoints, key=lambda x: x.epoch):
        table.add_row(
            checkpoint.name,
            str(checkpoint.epoch),
            f"{checkpoint.test_acc:.4f}",
            f"{checkpoint.avg_loss:.4f}",
            f"{checkpoint.lr:.2e}",
        )

    console.print(table)


@app.command(help="List all checkpoints (shortcut)")
def ls(name: str):
    list_checkpoints(name)


@app.command(help="Remove a checkpoint")
def remove(name: str):
    if not typer.confirm(f"Are you sure you want to remove {name} checkpoints?"):
        console.print("[bold red]Remove cancelled[/bold red]")
        return

    shutil.rmtree(CHECKPOINT_ROOT / name)
    console.print(f"[bold green]Checkpoints {name} removed[/bold green]")


@app.command(help="Remove a checkpoint (shortcut)")
def rm(name: str):
    remove(name)


if __name__ == "__main__":
    app()
