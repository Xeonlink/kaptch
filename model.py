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
    # rgba -> rgb
    img = img[:, :, :, :3]  # (1, H, W, 4) -> (1, H, W, 3)
    # uint8 -> float32
    img = img.float()
    # rgb -> gray
    img = img.mean(dim=3, keepdim=True)  # (1, H, W, 3) -> (1, H, W, 1)
    # normalize to [0, 1]
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


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


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
        # Encoder
        self.enc1 = UNetBlock(4, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = UNetBlock(64, 128)
        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(64, 32)
        # Feature aggregation
        # self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 64 * 256, 5 * 10)

    @staticmethod
    def preprocess_image(img: torch.Tensor) -> torch.Tensor:
        return preprocess2(img)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = F.pad(x, (0, target_W - W, 0, target_H - H), mode="reflect")

        # Encoder
        e1 = self.enc1(x)  # (B, 32, H, W)
        p1 = self.pool1(e1)  # (B, 32, H/2, W/2)
        e2 = self.enc2(p1)  # (B, 64, H/2, W/2)
        p2 = self.pool2(e2)  # (B, 64, H/4, W/4)
        # Bottleneck
        b = self.bottleneck(p2)  # (B, 128, H/4, W/4)
        # Decoder
        u2 = self.up2(b)  # (B, 64, H/2, W/2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))  # (B, 64, H/2, W/2)
        u1 = self.up1(d2)  # (B, 32, H, W)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))  # (B, 32, H, W)
        # Feature aggregation
        # x = self.dropout(d1)  # (B, 32, H, W)
        x = self.flatten(d1)  # (B, 32*H*W)
        x = self.fc(x)  # (B, 5*10)
        x = x.view(-1, 5, 10)  # (B, 5, 10)
        return x
