import torch
import torch.nn as nn
from pathlib import Path


# --------------------------------------------------
# 1. Define the same model architecture used in training
# --------------------------------------------------

class DrowsinessCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --------------------------------------------------
# 2. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "best_model.pth"
ONNX_PATH = ROOT / "models" / "drowsiness_cnn.onnx"


# --------------------------------------------------
# 3. Load trained model
# --------------------------------------------------

print("Loading trained model...")

model = DrowsinessCNN()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# 4. Create dummy input
# --------------------------------------------------

dummy_input = torch.randn(1, 1, 64, 64)


# --------------------------------------------------
# 5. Export to ONNX
# --------------------------------------------------

print("Exporting to ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamo=False
)

print("ONNX export completed.")
print(f"Saved to: {ONNX_PATH}")


# --------------------------------------------------
# 6. Check file
# --------------------------------------------------

if ONNX_PATH.exists():
    print(f"ONNX file size: {ONNX_PATH.stat().st_size / (1024 * 1024):.2f} MB")
else:
    print("ERROR: ONNX file was not created.")