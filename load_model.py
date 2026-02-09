from dataclasses import dataclass
from pathlib import Path

import torch
from torchinfo import summary

from backbones.mobilefacenet import MobileFaceNet

script_dir = Path(__file__).parent


@dataclass
class Config:
    model_fp = str(script_dir / "output" / "AdaDistill" / "MFN_AdaArcDistill_backbone.pth")
    input_size = (112, 112)
    embedding_size = 512


def main(args: Config):
    model = MobileFaceNet(input_size=args.input_size, embedding_size=args.embedding_size)
    weights = torch.load(args.model_fp, weights_only=True)
    model.load_state_dict(weights)

    summary(model, input_size=(1, 3, 112, 112))


if __name__ == "__main__":
    args = Config
    main(args)
