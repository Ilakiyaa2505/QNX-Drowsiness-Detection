import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path


# --------------------------------------------------
# 1. Define the same model architecture
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
        return self.classifier(x)


# --------------------------------------------------
# 2. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "best_model.pth"
ONNX_PATH = ROOT / "models" / "drowsiness_cnn.onnx"


# --------------------------------------------------
# 3. Check ONNX structure
# --------------------------------------------------

print("=" * 60)
print("ONNX MODEL VERIFICATION")
print("=" * 60)

print("\nChecking ONNX file...")

onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)

print("ONNX structure: OK")


# --------------------------------------------------
# 4. Load PyTorch model
# --------------------------------------------------

print("\nLoading PyTorch model...")

torch_model = DrowsinessCNN()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    torch_model.load_state_dict(checkpoint["model_state_dict"])
else:
    torch_model.load_state_dict(checkpoint)

torch_model.eval()

print("PyTorch model: OK")


# --------------------------------------------------
# 5. Create test input
# --------------------------------------------------

torch_input = torch.randn(1, 1, 64, 64)

print("\nTest input shape:", tuple(torch_input.shape))


# --------------------------------------------------
# 6. PyTorch inference
# --------------------------------------------------

with torch.no_grad():
    torch_output = torch_model(torch_input)

torch_output = torch_output.numpy()

print("PyTorch output:", torch_output)


# --------------------------------------------------
# 7. ONNX Runtime inference
# --------------------------------------------------

print("\nLoading ONNX Runtime...")

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("ONNX input name :", input_name)
print("ONNX output name:", output_name)

onnx_input = torch_input.numpy().astype(np.float32)

onnx_output = session.run(
    [output_name],
    {input_name: onnx_input}
)[0]

print("ONNX output:", onnx_output)


# --------------------------------------------------
# 8. Compare outputs
# --------------------------------------------------

difference = np.max(
    np.abs(torch_output - onnx_output)
)

print("\nMaximum output difference:", difference)


# --------------------------------------------------
# 9. Final result
# --------------------------------------------------

print("\n" + "=" * 60)

if difference < 1e-4:
    print("SUCCESS: PyTorch and ONNX outputs match.")
    print("The ONNX model is ready for the next stage.")
else:
    print("WARNING: Outputs differ more than expected.")
    print("Do not proceed until this is investigated.")

print("=" * 60)