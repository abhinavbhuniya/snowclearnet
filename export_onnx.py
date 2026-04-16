import torch
from src.model_arch import DATSRF

# Load model
model = DATSRF()
model.load_state_dict(torch.load("src/datsrf_model.pth", map_location="cpu"))
model.eval()

# Dummy input (must match your processing size = 128)
dummy = torch.randn(1, 3, 128, 128)

# Export
torch.onnx.export(
    model,
    dummy,
    "datsrf_model.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"]
)

print("ONNX model exported successfully")