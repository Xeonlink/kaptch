import cv2
import typer
import torch
from src.train.model import CRNNNet
from src.constants import BLANK_INDEX, CHECKPOINT_ROOT, DATASET_ROOT, DATA_CSV
import csv
import onnxruntime as ort
import numpy as np
from rich.console import Console
from pathlib import Path
from src.train.checkpoint import Checkpoint
from rich.panel import Panel
from typing_extensions import Annotated

console = Console()
app = typer.Typer(help="기타 유틸리티 도구", rich_markup_mode="rich")


def _load_image(img_path: str) -> torch.Tensor:
    """이미지를 로드하고 전처리합니다.

    Parameters:
        img_path (str): 이미지 파일 경로

    Returns:
        torch.Tensor: 전처리된 이미지 텐서 (W, H, 3)
    """
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = image.transpose(1, 0, 2)  # (H, W, 3) -> (W, H, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image)
    return image


def _decode(output: list[int]) -> str:
    """CTC 디코딩 결과를 문자열로 변환합니다.

    Parameters:
        output (list[int]): CTC 모델의 예측 결과

    Returns:
        str: 디코딩된 문자열
    """
    out = ""
    prev = -1
    for p in output:
        if p != prev and p != BLANK_INDEX:
            out += str(p)
        prev = p
    return out


@app.command(help="PyTorch 체크포인트를 ONNX 형식으로 변환합니다")
def torch2onnx(
    name: str,
    checkpoint_name: str,
    output: Annotated[str, typer.Option(help="출력 ONNX 파일명")] = "captcha.onnx",
    verbose: Annotated[bool, typer.Option(help="상세 출력 모드")] = False,
):
    """PyTorch 모델 체크포인트를 ONNX 형식으로 변환합니다.

    Parameters:
        name (str): 데이터셋 이름
        checkpoint_name (str): 체크포인트 파일명 (예: ep4.pkl)
        output (str): 출력 ONNX 파일명 (기본값: captcha.onnx)
        verbose (bool): 상세 출력 모드 (기본값: False)

    Examples:
        python -m src.misc torch2onnx sci ep4.pkl
        python -m src.misc torch2onnx nice ep10.pkl my-model.onnx
        python -m src.misc torch2onnx nhn_kcp ep5.pkl --verbose

    변환 전 확인사항:
    - 체크포인트 파일이 존재하는지 확인
    - 데이터셋의 첫 번째 이미지로 모의 입력 생성
    - 모델 구조와 파라미터 정보 표시

    변환된 ONNX 파일은 추론 시 더 빠른 속도를 제공합니다.
    """
    checkpoint_path = Path(CHECKPOINT_ROOT / name / checkpoint_name)
    checkpoint = Checkpoint.load(checkpoint_path)
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
            f"Checkpoint: [green]{name}/{checkpoint_name}[/]",
            f"Expected Size: [green]{checkpoint_path.stat().st_size / 1024:.2f} KB[/]",
            f"Parameter Count: [green]{sum(p.numel() for p in script_module.parameters()) / 1_000_000:.2f} M[/]",
            f"Input Dimensions: [green]{mock_input.shape}[/]",
            f"Input Data Type: [green]{mock_input.dtype}[/]",
            f"Output: [green]{output}[/]",
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
        console.print("Conversion cancelled.", style="red")
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

    console.print(f"\n[bold green]Conversion finished.[/]")


@app.command(help="ONNX 모델을 사용하여 이미지를 검증합니다", hidden=True)
def validate(
    name: str,
    checkpoint_name: str,
    image_path: Annotated[str, typer.Option(help="테스트할 이미지 파일 경로")],
):
    """ONNX 모델을 사용하여 특정 이미지에 대한 예측을 수행합니다.

    Parameters:
        name (str): 데이터셋 이름
        checkpoint_name (str): 체크포인트 파일명
        image_path (str): 테스트할 이미지 파일 경로

    Examples:
        python -m src.misc validate sci ep4.pkl test-image.png
        python -m src.misc validate nice ep10.pkl demo/captcha.png

    이 명령어는 개발 및 디버깅 목적으로 사용됩니다.
    """
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
    console.print(f"Output: [bold green]{output}[/]")


if __name__ == "__main__":
    app()
