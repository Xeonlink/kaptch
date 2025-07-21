import os
import cv2
import torch
import numpy as np


def main():
    path = os.path.join("dataset", "kmcert", "data", "000", "001.png")

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    image = image.float() / 255.0

    image = (image * 255).clamp(0, 255).to(torch.uint8)
    image = image.permute(1, 2, 0)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test.png", image)


if __name__ == "__main__":
    main()
