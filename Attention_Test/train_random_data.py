from TransModel import MultiSeqModel
import numpy as np
import matplotlib.pyplot as plt
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_prep_func import get_data_loader
from data_prep_func import get_device


device = get_device(gpu_index=1)

data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"
max_length, train_loader = get_data_loader(data_path,100)
# Set up the range for attention dimensions
attention_dims = range(20, 151, 10)  # Attention dimensions from 20 to 250, step 10

print(attention_dims)
# Initialize empty lists to store results

all_losses = []
all_accuracies = []
all_attention_dims = []

# Loop through different attention dimensions
for attention_dim in attention_dims:
    print(f"Training model with attention_dim={attention_dim}")
    
    # Initialize model with current attention dimension
    model = MultiSeqModel(input_length=max_length, tar_length=200, conv_1_dim=10, attention_dim=attention_dim)
    
    # Train model and get losses and accuracies
    losses, accuracies = model.train_model(train_loader, num_epochs=100, learning_rate=0.0005, device=device)
    
    # Store results
    all_losses.append(losses)
    all_accuracies.append(accuracies)
    all_attention_dims.append(attention_dim)

# Optionally, save the results to disk for later analysis (e.g., using pickle or numpy)

np.save('/workspaces/MoBa_FP/Experiments/Attention_Exp_0/all_losses.npy', all_losses)
np.save('/workspaces/MoBa_FP/Experiments/Attention_Exp_0/all_accuracies.npy', all_accuracies)
np.save('/workspaces/MoBa_FP/Experiments/Attention_Exp_0/all_attention_dims.npy', all_attention_dims)

print("Training of all models completed!")