import os
import cv2
import typer
import torch
from src.train.model import CRNNNet
from src.constants import BLANK_INDEX, CHECKPOINT_ROOT, DATASET_ROOT, DATA_CSV
import csv
import onnxruntime as ort
import numpy as np
from rich.tree import Tree
from rich.console import Console
from pathlib import Path
from src.train.checkpoint import Checkpoint
from rich.panel import Panel

console = Console()
app = typer.Typer()


def _load_image(img_path: str) -> torch.Tensor:
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = image.transpose(1, 0, 2)  # (H, W, 3) -> (W, H, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image)
    return image


def _decode(output: list[int]) -> str:
    out = ""
    prev = -1
    for p in output:
        if p != prev and p != BLANK_INDEX:
            out += str(p)
        prev = p
    return out


def _tree_from_path(folder_path: Path, tree: Tree) -> Tree:
    for path in folder_path.iterdir():
        if path.is_dir():
            subtree = tree.add(path.name)
            _tree_from_path(path, subtree)
        else:
            tree.add(path.name)
    return tree


@app.command(help="Convert a checkpoint to ONNX")
def torch2onnx(name: str, checkpoint_name: str, output: str = "captcha.onnx", verbose: bool = False):
    checkpoint = Checkpoint.load(CHECKPOINT_ROOT / name / checkpoint_name)
    model = CRNNNet()
    model.load_state_dict(checkpoint.state_dict)
    model.eval()
    script_module = torch.jit.script(model)

    with open(DATASET_ROOT / name / DATA_CSV) as f:
        reader = csv.reader(f)
        path, _ = next(reader)

        mock_input = _load_image(DATASET_ROOT / name / path)
        mock_input = mock_input.permute(2, 1, 0).unsqueeze(0).float() / 255.0

    panel_content = "\n".join(
        [
            f"Checkpoint: [bold green]{name}/{checkpoint_name}[/bold green]",
            f"Expected Size: [bold green]{os.path.getsize(output) / 1024:.2f} KB[/bold green]",
            f"Parameter Count: [bold green]{sum(p.numel() for p in script_module.parameters()) / 1000000:.2f} M[/bold green]",
            f"Input Dimensions: [bold green]{mock_input.shape}[/bold green]",
            f"Output Path: [bold green]{output}[/bold green]",
        ]
    )
    panel = Panel(
        panel_content,
        title=f"Conversion Details",
        title_align="left",
        border_style="green bold",
        padding=(1, 2),
    )
    console.print(panel)

    if not typer.confirm("Are you sure you want to convert the model to ONNX?"):
        console.print("Conversion cancelled.")
        return

    torch.onnx.export(
        model=script_module,
        args=(mock_input,),
        f=output,
        input_names=["x"],
        output_names=["y"],
        opset_version=11,
        dynamic_axes={
            "x": {
                # 0: "batch_size",
                2: "height",
                3: "width",
            },
            # "y": {0: "batch_size", 2: "height", 3: "width"},
        },
        verbose=verbose,
    )

    console.print(f"\n[bold green]Conversion finished.[/bold green]")


@app.command(help="Validate a model", hidden=True)
def validate(name: str, checkpoint_name: str, image_path: str):
    model_path = CHECKPOINT_ROOT / name / checkpoint_name
    model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    image = _load_image(image_path)
    image = image.permute(2, 1, 0).unsqueeze(0).float() / 255.0
    output = model.run(None, {"x": image.numpy()})
    output: np.ndarray = output[0]
    output = output.argmax(axis=2)
    output = output.tolist()
    output = output[0]
    output = _decode(output)
    console.print(f"Output: [bold green]{output}[/bold green]")


if __name__ == "__main__":
    name = "sci"
    checkpoint = "ep4.pkl"
    output = "captcha.onnx"
    torch2onnx(name, checkpoint, output, verbose=False)

# if __name__ == "__main__":
#     name = "sci"
#     checkpoint = "ep4.pkl"
#     image_path = "demo/sci.png"
#     validate(name, checkpoint, image_path)
