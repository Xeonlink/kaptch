import typer
import pyfiglet
from pathlib import Path
from src.datasets import app as datasets_app
from src.train import app as train_app
from src.misc import app as misc_app
from src.checkpoints import app as checkpoints_app

DATASET_ROOT = Path("dataset")

app = typer.Typer()
app.add_typer(datasets_app, name="datasets")
app.add_typer(train_app)
app.add_typer(checkpoints_app, name="checkpoints")
app.add_typer(misc_app, name="misc")


if __name__ == "__main__":
    pyfiglet.print_figlet("Kaptcha")
    app()
