import os
import cv2
import torch
from model import CRNNNet


def load_model(model_path: str) -> CRNNNet:
    model = CRNNNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_image(img_path: str) -> torch.Tensor:
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = image.transpose(1, 0, 2)  # (H, W, 3) -> (W, H, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image)
    return image


def main():
    model = load_model(os.path.join("checkpoints", "sci", "captcha_ep5.pkl"))
    script_module = torch.jit.script(model)

    mock_input = load_image(os.path.join("demo", "sci.png"))
    mock_input = mock_input.permute(2, 1, 0).unsqueeze(0).float() / 255.0

    torch.onnx.export(
        model=script_module,
        args=(mock_input,),
        f="captcha.onnx",
        input_names=["x"],
        output_names=["y"],
        opset_version=11,
        dynamic_axes={
            "x": {
                # 0: "batch_size",
                2: "height",
                3: "width",
            },
            # "y": {0: "batch_size", 2: "height", 3: "width"},
        },
    )


if __name__ == "__main__":
    main()
