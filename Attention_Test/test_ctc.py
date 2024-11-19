from TransCTCModel import MultiSeqCTCModel
import numpy as np
import matplotlib.pyplot as plt
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_prep_func import get_data_loader
from data_prep_func import get_device
from data_prep_func import vectors_to_sequence
from data_prep_func import decode_ctc_output
from Levenshtein import distance


device = get_device(gpu_index=2)


data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"
batch_sizes = [4,8,12,16,24,32,48,64]

loss_curves = {}

for b in batch_sizes:
    # Load data and initialize model
    max_length, train_loader = get_data_loader(data_path, 500, batch_size=b, dim_squeeze=True, num_reads=1)
    model = MultiSeqCTCModel(input_length=max_length, tar_length=200, conv_1_dim=16, conv_2_dim=48, attention_dim=64)
    
    # Train the model and store the loss curve
    losses, accuracies = model.train_model(train_loader, num_epochs=50, learning_rate=0.005, device=device)
    loss_curves[b] = losses  # Save the loss curve for this batch size

# Plot the loss curves
plt.figure(figsize=(10, 6))
for b, losses in loss_curves.items():
    plt.plot(losses, label=f'Batch Size: {b}')

# Customize the plot
plt.title("Loss Curves for Different Batch Sizes")
plt.xlabel("Epoch")
plt.ylabel("CTC Loss")
plt.legend()
plt.grid()
plt.show()

# Testing Collapse
#prob_sequence = [
#     [0.1, 0.2, 0.7],  # Highest prob -> index 2
#     [0.1, 0.2, 0.7],  # Highest prob -> index 2
#     [0.6, 0.3, 0.1],  # Highest prob -> index 0 (blank)
#     [0.2, 0.5, 0.3],  # Highest prob -> index 1
#     [0.2, 0.5, 0.3],  # Highest prob -> index 1
#     [0.7, 0.2, 0.1]   # Highest prob -> index 0 (blank)
# ]

# collapsed_output = model.ctc_collapse_probabilities(prob_sequence)
# print("Collapsed Output:", collapsed_output)