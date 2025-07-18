import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.fc = nn.Linear(16 * 8 * 20, 5 * 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (torch.Tensor): (batch, 4, H, W), 0~1, float32
        Returns:
            torch.Tensor: (batch, 5, 10) or list[str]
        """

        # 입력 이미지의 H, W가 32의 배수가 되도록 padding 추가
        B, C, H, W = x.shape
        ph = ((H - 1) // 32 + 1) * 32
        pw = ((W - 1) // 32 + 1) * 32
        padding = (0, pw - W, 0, ph - H)
        x = F.pad(x, padding)

        # 입력 이미지의 H, W가 2의 제곱수가 되도록 crop
        # B, C, H, W = x.shape
        # target_H = 1 if H == 0 else 2 ** ((H - 1).bit_length() - 1)
        # target_W = 1 if W == 0 else 2 ** ((W - 1).bit_length() - 1)
        # shrink_H = (H - target_H) // 2
        # shrink_W = (W - target_W) // 2
        # x = x[:, :, shrink_H : shrink_H + target_H, shrink_W : shrink_W + target_W]

        # 입력 이미지의 H, W가 2의 제곱수가 되도록 reflection padding 추가
        # B, C, H, W = x.shape
        # target_H = 1 if H == 0 else 2 ** (H - 1).bit_length()
        # target_W = 1 if W == 0 else 2 ** (W - 1).bit_length()
        # x = F.pad(x, (0, target_W - W, 0, target_H - H))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)

        x = self.dropout(x)  # (B, 32, H/2, W/2)
        x = self.flatten(x)  # (B, 32*H/2*W/2)
        x = self.fc(x)  # (B, 5*10)
        x = x.view(-1, 5, 10)  # (B, 5, 10)

        return x
