import torch
from solution import SWAInferenceHandler, InferenceMode

# Create handler with small dummy training tensor
h = SWAInferenceHandler(train_xs=torch.randn(2,3,60,60), model_dir='.')
# Use full SWAG
h.inference_mode = InferenceMode.SWAG_FULL
# Pretend we have seen at least one snapshot
h.num_snapshots = 2

# Populate sum_weights and sum_sq_weights and add small deviations for each parameter
for name, param in h.network.named_parameters():
    h.sum_weights[name] = param.detach().clone()
    h.sum_sq_weights[name] = (param.detach().clone())**2
    # Add a couple of small deviations to the deque
    h.deviation_matrix[name].append(torch.randn_like(param) * 0.01)
    h.deviation_matrix[name].append(torch.randn_like(param) * 0.02)

# Call sampling routine which previously raised due to .t() on >2D tensor
h.sample_parameters()
print('sample_parameters completed ok')
