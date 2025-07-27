import os
import csv
import typer
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.train.dataset import CaptchaDataset
from src.train.model import CRNNNet
from rich.console import Console
from rich.panel import Panel
from src.train.checkpoint import Checkpoint
from src.constants import CHECKPOINT_ROOT, DATASET_ROOT, DATA_CSV

console = Console()
app = typer.Typer(help="모델 훈련 및 평가 도구", rich_markup_mode="rich")


def get_device() -> torch.device:
    """사용 가능한 torch device를 반환합니다.

    Returns:
        torch.device: CUDA > MPS > CPU 순서로 사용 가능한 디바이스
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_labels_ctc(labels: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters:
        labels (list[str]): 각 샘플의 숫자 문자열 (batch)
    Returns:
        targets (torch.Tensor): (sum_label_len,) 모든 라벨을 1D로 이어붙인 int tensor
        lengths (torch.Tensor): (batch,) 각 라벨의 길이
    """
    targets = [int(ch) for label in labels for ch in label]
    lengths = [len(label) for label in labels]
    return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def ctc_greedy_decode(logits: torch.Tensor) -> list[list[int]]:
    """CTC 디코딩: 중복과 blank 토큰 제거"""
    pred = logits.argmax(dim=2).cpu().numpy()  # (B, T)
    results = []
    for seq in pred:
        out = []
        prev = -1
        for p in seq:
            if p != prev and p != 10:  # 10: blank
                out.append(p)
            prev = p
        results.append(out)
    return results


def evaluate_ctc(model: CRNNNet, loader: DataLoader, device: torch.device) -> float:
    """CRNN 모델 평가"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model.forward(imgs)
            preds = ctc_greedy_decode(logits)
            labels_num = [[int(ch) for ch in label] for label in labels]
            for p, t in zip(preds, labels_num):
                if p == t:
                    correct += 1
                total += 1
    return correct / total if total > 0 else 0.0


def is_trainable(dataset_name: str) -> bool:
    """데이터셋이 훈련 가능한 상태인지 검증합니다.

    Parameters:
        dataset_name (str): 검증할 데이터셋 이름

    Returns:
        bool: 훈련 가능하면 True, 그렇지 않으면 False
    """
    if not (DATASET_ROOT / dataset_name).exists():
        console.print(f"[bold red]Dataset {dataset_name} not found[/]")
        return False

    if not (DATASET_ROOT / dataset_name / DATA_CSV).exists():
        console.print(f"[bold red]Dataset {dataset_name} is not labeled[/]")
        return False

    with open(DATASET_ROOT / dataset_name / DATA_CSV, "r") as f:
        reader = csv.reader(f)
        for path, label in reader:
            if not (DATASET_ROOT / dataset_name / path).exists():
                console.print(f"Image [bold red]not exist at {path}[/]")
                return False

            if label == "":
                console.print(f"Image [bold red]has no label at {path}[/]")
                return False

    return True


@app.command(help="CRNN 모델을 훈련합니다")
def train(
    name: str,
    # 하이퍼파라미터
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 2e-3,
    patience: int = 5,
    train_size: int = 1_000,
    test_size: int = 100,
    seed: int = 42,
    warmup_epochs: int = 5,
):
    """CRNN 모델을 사용하여 캡챠 인식 모델을 훈련합니다.

    Parameters:
        name (str): 훈련할 데이터셋 이름
        batch_size (int): 배치 크기 (기본값: 32)
        epochs (int): 훈련 에포크 수 (기본값: 50)
        learning_rate (float): 학습률 (기본값: 2e-3)
        patience (int): Early stopping 인내심 (기본값: 5)
        train_size (int): 훈련 데이터 크기 (기본값: 1000)
        test_size (int): 테스트 데이터 크기 (기본값: 100)
        seed (int): 랜덤 시드 (기본값: 42)
        warmup_epochs (int): 워밍업 에포크 수 (기본값: 5)

    Examples:
        python -m src.train train sci
        python -m src.train train nice --epochs 100 --batch-size 64
        python -m src.train train nhn_kcp --learning-rate 1e-3 --patience 10

    훈련 전 확인사항:
    - 데이터셋이 존재하는지 확인
    - 모든 이미지에 라벨이 있는지 확인
    - 충분한 GPU 메모리가 있는지 확인

    훈련 중 체크포인트는 checkpoints/{name}/ 폴더에 저장됩니다.
    """
    if not is_trainable(name):
        return

    panel_content = "".join(
        [
            f"Batch size: [green]{batch_size}[/]\n",
            f"Epochs: [green]{epochs}[/]\n",
            f"Learning rate: [green]{learning_rate:.0e}[/]\n",
            f"Patience: [green]{patience}[/]\n",
            f"Train size: [green]{train_size}[/]\n",
            f"Test size: [green]{test_size}[/]\n",
            "\n",
            f"[red]This will remove all checkpoints and start training from scratch.[/]",
        ]
    )
    panel = Panel(
        panel_content,
        title=f"[bold green]Traning {name}[/]",
        title_align="left",
        border_style="bold green",
        padding=(1, 2),
    )
    console.print(panel)

    if not typer.confirm(f"Are you sure you want to train {name}?"):
        console.print("[bold red]Training cancelled[/]")
        return

    device = get_device()
    shutil.rmtree(CHECKPOINT_ROOT / name, ignore_errors=True)
    os.makedirs(CHECKPOINT_ROOT / name, exist_ok=True)

    # dataset
    dataset = CaptchaDataset(DATASET_ROOT / name)
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [train_size, test_size], generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # model, loss function, optimizer
    model = CRNNNet(num_classes=11).to(device)
    criterion = nn.CTCLoss(blank=10, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)

    best_acc = 0.0
    patience_counter = 0

    console.print("")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            targets, target_lengths = encode_labels_ctc(labels)
            logits = model.forward(imgs)  # (B, T, C)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            input_lengths = torch.full((imgs.size(0),), logits.size(1), dtype=torch.long)

            # Move tensors to CPU for CTC loss (MPS doesn't support CTC)
            log_probs_cpu = log_probs.cpu()
            loss = criterion.forward(log_probs_cpu, targets, input_lengths, target_lengths)
            # Move loss back to device for backward pass
            loss = loss.to(device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        acc = evaluate_ctc(model, test_loader, device)

        # save checkpoint
        checkpoint = Checkpoint(model.state_dict(), epoch=epoch, test_acc=acc, avg_loss=avg_loss, lr=current_lr)
        checkpoint.save(CHECKPOINT_ROOT / name / f"ep{epoch}.pkl")

        # print progress
        if acc > best_acc:
            console.print(
                f"[default not bold]Epoch: {epoch:02d} | Test Acc: [green]{acc:.4f}[/] | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}[/]"
            )
        elif acc == best_acc:
            console.print(
                f"[default not bold]Epoch: {epoch:02d} | Test Acc: [yellow]{acc:.4f}[/] | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}[/]"
            )
        else:
            console.print(
                f"[default not bold]Epoch: {epoch:02d} | Test Acc: [red]{acc:.4f}[/] | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}[/]"
            )

        # Early stopping logic
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                console.print("")
                console.print(f"Early stopped at [yellow]{epoch}[/] epoch (patience: [yellow]{patience}[/] epochs)")
                console.print(f"Training finished. Final test acc: {acc:.4f}")
                return

    # finish training
    console.print(f"Training finished. Final test acc: [bold green]{acc:.4f}[/]")


if __name__ == "__main__":
    train()
