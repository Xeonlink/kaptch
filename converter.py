import cv2
import torch
from model import CaptchaNet


def load_model(model_path: str) -> CaptchaNet:
    model = CaptchaNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_image(img_path: str) -> torch.Tensor:
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = image.transpose(1, 0, 2)  # (H, W, 4) -> (W, H, 4)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    image = torch.from_numpy(image)
    return image


def main():
    model = load_model("checkpoints/captcha_ep12.pkl")
    script_module = torch.jit.script(model)

    mock_input = load_image("demo/0000.png")
    mock_input = mock_input.permute(2, 1, 0).unsqueeze(0).float()

    torch.onnx.export(
        model=script_module,
        args=(mock_input,),
        f="captcha.onnx",
        input_names=["x"],
        output_names=["y"],
        opset_version=10,
        # dynamic_axes={
        #     "x": {0: "batch_size", 2: "height", 3: "width"},
        #     "y": {0: "batch_size", 2: "height", 3: "width"},
        # },
    )


if __name__ == "__main__":
    main()
