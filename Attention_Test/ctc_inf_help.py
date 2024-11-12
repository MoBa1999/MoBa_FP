import torch.nn as nn
import torch
import os
import torch.nn.functional as F

import numpy as np

# Assuming your log_probs data is a NumPy array
log_probs_data = np.array([
    [[ 1.5597e-03, -1.4458e-03, -7.8225e-03,  7.3491e-03,  1.8900e-02]],
         [[ 1.4176e-02,  3.9799e-03,  8.9428e-03,  8.3385e-03, -1.5126e-02]],
         [[-1.0706e-02,  8.0439e-03, -6.2857e-03,  1.8705e-03,  6.7295e-03]],
         [[ 1.5988e-03, -9.1409e-03, -7.6503e-03,  1.0288e-02, -1.5170e-02]],
         [[-1.2602e-02,  1.5634e-02, -2.4396e-02,  1.4990e-02, -1.3542e-03]]
])


log_probs_data = np.array([
    [[-1.9, -0.5, -1.8, -1.6, -2.2]],
    [[-1.7, -0.4, -1.9, -1.8, -2.2]],
    [[-1.6, -0.1, -1.7, -1.9, -1.1]],
    [[-1.8, -0.3, -1.3, -1.3, -1.6]],
    [[-0.2, -1.2, -0.3, -1.5, -1.8]],
    [[-0.2, -1.2, -1.9, -1.5, -1.2]],

])

log_probs = torch.tensor(log_probs_data).float()

# Targets
targets = torch.tensor([
    [4, 4, 1, 2, 4, 1]]).float()



input_lengths = torch.tensor([6])  # Assuming 200 time steps for each sample
target_lengths = torch.tensor([6])  # Assuming 200 target characters for each sample

# Calculate CTC Loss
loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
print(loss)