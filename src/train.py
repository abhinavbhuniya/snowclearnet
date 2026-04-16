import torch
from model_arch import DATSRF

model = DATSRF()

# Save random weights (for demo)
torch.save(model.state_dict(), "datsrf_model.pth")