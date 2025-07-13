import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import CaptchaDataset
from model import CaptchaNet


def get_device() -> torch.device:
    """
    사용 가능한 torch device를 반환하는 함수.
    Parameters:
        None
    Returns:
        torch.device: 사용 가능한 device (cuda > mps > cpu 순서)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# 하이퍼파라미터
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
LOG_INTERVAL = 100
DEVICE = get_device()
IMG_HEIGHT = 80
IMG_WIDTH = 200
NUM_CLASSES = 10
NUM_DIGITS = 5
PATIENCE = 5

# 디렉토리 생성
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)


def one_hot_labels(labels: list[str]) -> torch.Tensor:
    """
    Parameters:
        labels (list[str]): 5자리 숫자 문자열 리스트 (batch)
    Returns:
        torch.Tensor: (batch, 5, 10) one-hot
    """
    label_tensor = torch.tensor([[int(ch) for ch in label] for label in labels])  # (batch, 5)
    one_hot = torch.nn.functional.one_hot(label_tensor, num_classes=NUM_CLASSES)  # (batch, 5, 10)
    return one_hot.float()


def evaluate(model: CaptchaNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels_oh = one_hot_labels(labels).to(device)
            outputs = model.forward(imgs)
            preds = outputs.argmax(dim=2)  # (batch, 5)
            labels_num = torch.tensor([[int(ch) for ch in label] for label in labels], device=device)
            correct += (preds == labels_num).all(dim=1).sum().item()
            total += imgs.size(0)
    return correct / total if total > 0 else 0.0


def main():
    # 데이터셋
    train_set = CaptchaDataset("train", "dataset")
    test_set = CaptchaDataset("test", "dataset")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 모델
    model = CaptchaNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            labels_oh = one_hot_labels(labels).to(DEVICE)

            outputs = model.forward(imgs)
            loss = criterion.forward(outputs, labels_oh)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + batch_idx)
        scheduler.step()

        acc = evaluate(model, test_loader, DEVICE)
        writer.add_scalar("Accuracy/test", acc, epoch)

        # 체크포인트 저장
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"captcha_ep{epoch}.pkl")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Epoch {epoch}] test_acc={acc:.4f}, checkpoint saved: {ckpt_path}")

        # Early stopping logic
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                writer.add_scalar("Accuracy/final_test", acc, epoch)
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "captcha_final.pt"))
                print(f"Training finished. Final test acc: {acc:.4f}")
                writer.close()
                return

    # 마지막 평가 및 저장
    acc = evaluate(model, test_loader, DEVICE)
    writer.add_scalar("Accuracy/final_test", acc, EPOCHS)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "captcha_final.pt"))
    print(f"Training finished. Final test acc: {acc:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
