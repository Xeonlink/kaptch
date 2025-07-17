"""
한국모바일인증1 데이터셋 이미지를 흰색 도화지에 올려놓았다고 가정하여,
투명한 부분을 제거하는 스크립트
"""

import os
import cv2
import numpy as np

DATASET_DIR = os.path.join("dataset", "한국모바일인증1")


def main() -> None:
    for file in os.listdir(DATASET_DIR):
        file_path = os.path.join(DATASET_DIR, file)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.shape[2] == 4:
            b, g, r, a = cv2.split(img)
            alpha = a.astype(float) / 255.0
            # 흰색 배경 생성
            white_bg = np.ones_like(img[:, :, :3], dtype=np.uint8) * 255
            # 알파 채널을 이용해 합성
            for c in range(3):
                img[:, :, c] = (
                    img[:, :, c].astype(float) * alpha + white_bg[:, :, c].astype(float) * (1 - alpha)
                ).astype(np.uint8)
            img = img[:, :, :3]  # BGR 3채널로 변환
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if img.shape[2] == 4 else img
        cv2.imwrite(file_path, img)


if __name__ == "__main__":
    main()
