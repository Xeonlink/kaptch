import torch
import torch.nn as nn
import torch.nn.functional as F


def preprocess1(img: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
        img (torch.Tensor): (W, H, 4) 또는 (batch, W, H, 4)
    Returns:
        torch.Tensor: (batch, 1, H, W) (grayscale)
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)  # (H, W, 4) -> (1, H, W, 4)

    mask = img[:, :, :, 3] < 255
    img[mask] = torch.tensor([255, 255, 255, 255], device=img.device, dtype=torch.uint8)
    img = img[:, :, :, :3]  # (1, H, W, 4) -> (1, H, W, 3)
    img = img.float()
    img = img.mean(dim=3, keepdim=True)  # (1, H, W, 3) -> (1, H, W, 1)
    img = img / 255.0
    img = img.permute(0, 3, 1, 2)  # (1, H, W, 1) -> (1, 1, H, W)
    return img


def preprocess2(img: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
        img (torch.Tensor): (W, H, 4) 또는 (batch, W, H, 4)
    Returns:
        torch.Tensor: (batch, 4, H, W) (rgba)
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)  # (W, H, 4) -> (1, W, H, 4)

    img = img.permute(0, 3, 2, 1)  # (1, W, H, 4) -> (1, 4, H, W)
    # uint8 -> float32
    img = img.float()
    # normalize to [0, 1]
    img = img / 255.0
    return img


class CaptchaNet(nn.Module):
    """
    캡챠 숫자 5자리를 예측하는 U-Net 기반 CNN 모델.
    Parameters:
        입력: (W, H, 4) 또는 (batch, W, H, 4)
    Returns:
        (batch, 5, 10): 각 자리별 0~9 one-hot 예측값
    """

    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 8 * 32, 5 * 10)

    @staticmethod
    def preprocess_image(img: torch.Tensor) -> torch.Tensor:
        return preprocess2(img)

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        """
        Parameters:
            x (torch.Tensor): (W, H, 4) 또는 (batch, W, H, 4)
        Returns:
            torch.Tensor: (batch, 5, 10)
        """
        x = self.preprocess_image(x)

        # 입력 이미지의 H, W가 2의 제곱수가 되도록 crop
        # B, C, H, W = x.shape
        # target_H = 1 if H == 0 else 2 ** ((H - 1).bit_length() - 1)
        # target_W = 1 if W == 0 else 2 ** ((W - 1).bit_length() - 1)
        # x = x[:, :, :target_H, :target_W]

        # 입력 이미지의 H, W가 2의 제곱수가 되도록 reflection padding 추가
        B, C, H, W = x.shape
        target_H = 1 if H == 0 else 2 ** (H - 1).bit_length()
        target_W = 1 if W == 0 else 2 ** (W - 1).bit_length()
        x = F.pad(x, (0, target_W - W, 0, target_H - H))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)

        x = self.dropout(x)  # (B, 32, H/2, W/2)
        x = self.flatten(x)  # (B, 32*H/2*W/2)
        x = self.fc(x)  # (B, 5*10)
        x = x.view(-1, 5, 10)  # (B, 5, 10)

        if not train:
            preds: list[list[int]] = x.argmax(dim=2).tolist()
            preds = ["".join(map(str, pred)) for pred in preds]
            return preds

        return x
