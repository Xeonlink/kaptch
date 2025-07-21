import os
import cv2
import onnxruntime as ort
import torch
import numpy as np


def load_onnx_model(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def load_image(image_path: str) -> torch.Tensor:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = image.transpose(1, 0, 2)  # (H, W, 3) -> (W, H, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image)
    return image


def decode(output: list[int]) -> str:
    out = ""
    prev = -1
    for p in output:
        if p != prev and p != 10:  # 10: blank
            out += str(p)
        prev = p
    return out


def main():

    model_path = os.path.join("captcha.onnx")
    model = load_onnx_model(model_path)

    image_path = os.path.join("demo", "nhnkcp.png")
    image = load_image(image_path)
    # image = torch.randint(0, 1, image.shape, dtype=torch.uint8)
    image = image.permute(2, 1, 0).unsqueeze(0).float() / 255.0
    output = model.run(None, {"x": image.numpy()})
    output: np.ndarray = output[0]
    output = output.argmax(axis=2)
    output = output.tolist()
    output = output[0]
    output = decode(output)
    print(output)


if __name__ == "__main__":
    main()
