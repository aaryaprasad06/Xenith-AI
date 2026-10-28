"""Export the AdaFace IR-101 weights to ONNX so `AdaFaceEmbedder` can run them in OpenCV.

    uv run python -m face.models adaface_ir101_weights
    uv run --group export python -m face.adaface_export

The network below is the IR-101 backbone from CVLFace (MIT), cut down to what
the WebFace12M checkpoint uses, so the weights load without running the
repository's own code. The export is checked against PyTorch before it is kept.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

from face.models import EXPORTED, MODELS_DIR, model_file

# (in_channels, out_channels, units) per stage of IR-101; each stage starts with stride 2.
IR101_STAGES = [(64, 64, 3), (64, 128, 13), (128, 256, 30), (256, 512, 3)]


class BasicBlockIR(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_layer(x) + self.shortcut_layer(x)


class IR101(nn.Module):
    def __init__(self, output_dim: int = 512) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64)
        )
        blocks = []
        for in_channel, depth, units in IR101_STAGES:
            blocks.append(BasicBlockIR(in_channel, depth, 2))
            blocks.extend(BasicBlockIR(depth, depth, 1) for _ in range(units - 1))
        self.body = nn.Sequential(*blocks)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, output_dim),
            nn.BatchNorm1d(output_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.body(self.input_layer(x)))


def load_ir101(weights: Path) -> IR101:
    """IR-101 with the CVLFace checkpoint loaded strictly (every key must match)."""
    state = load_file(str(weights))
    prefix = "model.net."
    state = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    net = IR101()
    net.load_state_dict(state, strict=True)
    return net.eval()


def export(weights: Path, out: Path) -> float:
    """Write the ONNX file and return the largest difference from PyTorch on test inputs."""
    net = load_ir101(weights)
    torch.onnx.export(
        net,
        (torch.zeros(1, 3, 112, 112),),
        str(out),
        input_names=["input"],
        output_names=["embedding"],
        dynamo=False,
    )
    check = cv2.dnn.readNetFromONNX(str(out))
    worst = 0.0
    for seed in range(3):
        x = torch.from_numpy(
            np.random.default_rng(seed)
            .uniform(-1, 1, (1, 3, 112, 112))
            .astype(np.float32)
        )
        with torch.no_grad():
            expected = net(x).numpy()
        check.setInput(x.numpy())
        actual = check.forward()
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError("OpenCV or PyTorch output contains non-finite values")
        worst = max(worst, float(np.abs(actual - expected).max()))
    return worst


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, default=MODELS_DIR / EXPORTED["adaface_ir101"][0]
    )
    args = parser.parse_args(argv)
    tmp = args.out.with_suffix(".tmp.onnx")
    try:
        worst = export(model_file("adaface_ir101_weights"), tmp)
        if not np.isfinite(worst) or worst > 1e-3:
            raise SystemExit(
                f"OpenCV output differs from PyTorch by {worst:.2e}; export not kept"
            )
        tmp.replace(args.out)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"{args.out} (largest difference from PyTorch {worst:.2e})")


if __name__ == "__main__":
    main()
