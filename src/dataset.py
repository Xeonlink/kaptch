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

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        csv_path = os.path.join(root_dir, "data_list.csv")
        self.samples: list[list[str]] = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            self.samples = list(reader)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """
        이미지와 라벨을 반환.
        Parameters:
            idx (int): 인덱스
        Returns:
            image: (w, h, 4) (rgba)
            label: 이미지에 적힌 숫자 (str)
        """

        rel_path, raw_label = self.samples[idx]
        img_path = os.path.join(self.root_dir, rel_path)

        # 이미지 유효성검사
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = image.transpose(1, 0, 2)  # (H, W, 4) -> (W, H, 4)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 라벨 유효성검사
        label = raw_label
        if not self._validate_label(label):
            raise ValueError(f"Invalid label: {label} (must be 5-digit string), {img_path}")

        # 이미지 변환 및 유효성검사
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        image = torch.from_numpy(image)
        if image.dtype != torch.uint8:
            raise ValueError(f"Image dtype must be uint8, but got {image.dtype}, {img_path}")
        if image.ndim != 3:
            raise ValueError(f"Image must have 3 dimensions, but got {image.ndim}, {img_path}")
        if image.shape[2] != 4:
            raise ValueError(f"Image must have 4 channels, but got {image.shape[2]}, {img_path}")

        return image, label

    def _validate_label(self, label: str) -> bool:
        """
        라벨이 5자리 숫자 문자열인지 검사.
        Parameters:
            label (str): 검사할 라벨
        Returns:
            bool: 유효하면 True, 아니면 False
        """
        return len(label) == 5 and label.isdigit()
