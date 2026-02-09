from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from torchinfo import summary

from backbones.mobilefacenet import MobileFaceNet

script_dir = Path(__file__).parent


@dataclass
class Config:
    model_fp = str(script_dir / "output" / "AdaDistill" / "MFN_AdaArcDistill_backbone.pth")
    onnx_fp = str(script_dir / "output" / "AdaDistill" / "mfn_adadistill.onnx")
    img_fp = str(script_dir / "data" / "00001.jpg")
    input_size = (112, 112)
    embedding_size = 512
    dynamic_axes = False


def export_model(args: Config):
    model = MobileFaceNet(input_size=args.input_size, embedding_size=args.embedding_size)
    weights = torch.load(args.model_fp, weights_only=True, map_location="cpu")
    model.load_state_dict(weights)
    model.eval()

    # summary(model, input_size=(1, 3, 112, 112))

    # Export model to onnx
    input_tensor = torch.rand((1, 3, 112, 112), dtype=torch.float32)

    output_fp = Path(args.model_fp).parent / "mfn_adadistill.onnx"
    torch.onnx.export(
        model,
        (input_tensor,),
        str(output_fp),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}} if args.dynamic_axes else None,
    )


def load_onnx(args: Config):
    img = cv2.imread(args.img_fp)
    print("Original image shape:", img.shape)

    model = cv2.dnn.readNetFromONNX(args.onnx_fp)

    blob = cv2.dnn.blobFromImage(
        img,
        1 / 127.5,
        (112, 112),
        127.5,  # Substract mean to normalize to range (-1, 1)
        swapRB=True,
        crop=False,
    )
    print("Blob shape:", blob.shape)

    # Set input and run inference
    model.setInput(blob)
    outputs = model.forward()
    print("ONNX output shape:", outputs.shape)


if __name__ == "__main__":
    args = Config
    # export_model(args)
    load_onnx(args)
