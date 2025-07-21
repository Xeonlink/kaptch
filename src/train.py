import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CaptchaDataset
from model import CRNNNet


def get_device() -> torch.device:
    """사용 가능한 torch device를 반환하는 함수."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 2e-3
CHECKPOINT_DIR = "checkpoints"
DEVICE = get_device()
PATIENCE = 5
TRAIN_SIZE = 1_000
TEST_SIZE = 100
SEED = 42
DATASET_ROOT = os.path.join("dataset", "sci")
WARMUP_EPOCHS = 5

# 디렉토리 생성
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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


def main():
    # 데이터셋
    dataset = CaptchaDataset(DATASET_ROOT)
    generator = torch.Generator().manual_seed(SEED)
    train_set, test_set = random_split(dataset, [TRAIN_SIZE, TEST_SIZE], generator)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 모델, 손실함수, 최적화기
    model = CRNNNet(num_classes=11).to(DEVICE)
    criterion = nn.CTCLoss(blank=10, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            targets, target_lengths = encode_labels_ctc(labels)
            logits = model.forward(imgs)  # (B, T, C)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            input_lengths = torch.full((imgs.size(0),), logits.size(1), dtype=torch.long)

            # Move tensors to CPU for CTC loss (MPS doesn't support CTC)
            log_probs_cpu = log_probs.cpu()
            loss = criterion.forward(log_probs_cpu, targets, input_lengths, target_lengths)
            # Move loss back to device for backward pass
            loss = loss.to(DEVICE)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Learning rate scheduling
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step()

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        acc = evaluate_ctc(model, test_loader, DEVICE)

        # 체크포인트 저장
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"captcha_ep{epoch}.pkl")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Epoch {epoch:3d}] test_acc={acc:.4f}, avg_loss={avg_loss:.4f}, lr={current_lr:.2e}")

        # Early stopping logic
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "captcha_final.pt"))
                print(f"Training finished. Final test acc: {acc:.4f}")
                return

    # 마지막 평가 및 저장
    acc = evaluate_ctc(model, test_loader, DEVICE)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "captcha_final.pt"))
    print(f"Training finished. Final test acc: {acc:.4f}")


if __name__ == "__main__":
    main()
