import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNNNet(nn.Module):
    """
    Optimized CRNN 기반 캡챠 인식 모델 (CTC loss용)
    Parameters:
        입력: (batch, 3, H, W), 0~1, float32
    Returns:
        (batch, T, num_classes): 각 time step별 0~9+blank 예측값 (logits)
    """

    def __init__(self, num_classes: int = 11, rnn_hidden: int = 32):
        super().__init__()

        # Efficient CNN with depthwise separable convolutions
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, 1, 1),  # (B, 32, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 32, H/2, W/2)
            # Block 2 - Depthwise separable
            nn.Conv2d(32, 32, 3, 1, 1, groups=32),  # Depthwise
            nn.Conv2d(32, 64, 1, 1, 0),  # Pointwise
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 64, H/4, W/4)
            # Block 3 - Depthwise separable
            nn.Conv2d(64, 64, 3, 1, 1, groups=64),  # Depthwise
            nn.Conv2d(64, 128, 1, 1, 0),  # Pointwise
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # (B, 128, H/8, W/4)
            # Block 4 - Lightweight
            nn.Conv2d(128, 128, 3, 1, 1, groups=128),  # Depthwise
            nn.Conv2d(128, 128, 1, 1, 0),  # Pointwise
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # (B, 128, H/16, W/4)
            nn.Dropout2d(0.2),
        )

        # Lightweight RNN
        # Input images are (3, 64<=H<80, W) -> CNN output is (128, 4, W/4) -> RNN input is 128*4=512
        self.rnn = nn.GRU(
            input_size=128 * 4,  # 128 channels * 4 height after pooling
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(rnn_hidden * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (torch.Tensor): (batch, 3, H, W), 0~1, float32
        Returns:
            torch.Tensor: (batch, T, num_classes) (logits)
        """
        # 입력 이미지의 H, W가 32, 32 이상인지 확인 (CTC 구조상)
        B, C, H, W = x.shape
        if H < 64:
            x = F.pad(x, (0, 0, 0, 64 - H), mode="constant", value=1.0)
        elif 80 <= H:
            x = F.interpolate(x, size=(79, W), mode="bilinear", align_corners=False)

        x = self.cnn(x)  # (B, 128, 4, W/4) for input (B, 3, 64<=H<80, W)
        x = x.permute(0, 3, 1, 2)  # (B, W/4, 128, 4)
        B, T, C, H_ = x.shape
        x = x.reshape(B, T, C * H_)  # (B, W/4, 512)

        x, _ = self.rnn(x)  # (B, W/4, rnn_hidden*2)
        x = self.fc(x)  # (B, W/4, num_classes)
        return x
