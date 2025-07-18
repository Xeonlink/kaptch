import cv2
import torch
from model import CaptchaNet


def load_image(img_path: str) -> torch.Tensor:
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = image.transpose(1, 0, 2)  # (H, W, 4) -> (W, H, 4)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    image = torch.from_numpy(image)
    return image


def load_model(model_path: str) -> CaptchaNet:
    model = CaptchaNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def main():
    model = load_model("checkpoints/captcha_ep40.pkl")
    for i in range(10):
        image_path = f"demo/{i:04d}.png"
        image = load_image(image_path)
        output = model.forward(image)
        print(f"{image_path}: {output.argmax(dim=2).squeeze(0).tolist()}")


if __name__ == "__main__":
    main()
