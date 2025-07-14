import cv2
import onnxruntime as ort
import torch
import onnx
import numpy as np


def load_onnx_model(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def load_image(image_path: str) -> torch.Tensor:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = image.transpose(1, 0, 2)  # (H, W, 4) -> (W, H, 4)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    image = torch.from_numpy(image)
    return image


def main():

    model = load_onnx_model("captcha.onnx")
    for i in range(10):
        image_path = f"demo/{i:04d}.png"
        image = load_image(image_path)
        image = image.permute(2, 1, 0).unsqueeze(0).float()
        output = model.run(None, {"x": image.numpy()})
        output: np.ndarray = output[0]
        output = output.argmax(axis=2)
        output = output.tolist()
        output = output[0]
        output = "".join(map(str, output))
        print(output)


if __name__ == "__main__":
    main()
