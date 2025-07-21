import os
import csv
import cv2
from torch.utils.data import Dataset
import torch


class CaptchaDataset(Dataset):
    """
    캡챠 이미지와 라벨을 반환하는 PyTorch Dataset 클래스.
    Parameters:
        root_dir: dataset 폴더의 경로 (csv와 이미지가 포함된 루트)
    """

    root_dir: str
    samples: list[list[str]]

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.samples = self._load_samples(os.path.join(root_dir, "data_list.csv"))

    def _load_samples(self, csv_path: str) -> list[list[str]]:
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            return list(reader)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """
        이미지와 라벨을 반환.
        Parameters:
            idx (int): 인덱스
        Returns:
            image: (3, h, w) (rgb), float32, 0.0~1.0
            label: 이미지에 적힌 숫자 (str)
        """

        rel_path, raw_label = self.samples[idx]
        img_path = os.path.join(self.root_dir, rel_path)

        # 이미지 유효성검사
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # (H, W, 3) bgr
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 라벨 유효성검사
        label = raw_label
        if not label.isdigit():
            raise ValueError(f"Invalid label: {label} (must be digit string), {img_path}")

        # 텐서로 변환 및 유효성검사
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image)
        if image.dtype != torch.uint8:
            raise ValueError(f"Image dtype must be uint8, but got {image.dtype}, {img_path}")
        if image.ndim != 3:
            raise ValueError(f"Image must have 3 dimensions, but got {image.ndim}, {img_path}")
        if image.shape[2] != 3:
            raise ValueError(f"Image must have 3 channels, but got {image.shape[2]}, {img_path}")

        # 이미지 정규화
        image = image.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        image = image.float() / 255.0  # (3, H, W) -> (3, H, W), 0.0~1.0

        return image, label
