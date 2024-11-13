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
device = None

data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"
max_length, train_loader = get_data_loader(data_path,batch_size=4)
# Set up the range for attention dimensions
heads  = [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36]  # Attention dimensions from 20 to 250, step 10

print(heads)
# Initialize empty lists to store results

all_losses = []
all_accuracies = []
all_heads = []

# Loop through different attention dimensions
for head in heads:
    print(f"Training n_heads={head}")
    
    # Initialize model with current attention dimension
    model = MultiSeqModel(input_length=max_length, tar_length=200, conv_1_dim=10,attention_dim=72,n_heads=head)
    
    # Train model and get losses and accuracies
    losses, accuracies = model.train_model(train_loader, num_epochs=100, learning_rate=0.001, device=device)
    
    # Store results
    all_losses.append(losses)
    all_accuracies.append(accuracies)
    all_heads.append(head)

# Optionally, save the results to disk for later analysis (e.g., using pickle or numpy)

np.save('/workspaces/MoBa_FP/Experiments/Attention_Exp_1/all_losses.npy', all_losses)
np.save('/workspaces/MoBa_FP/Experiments/Attention_Exp_1/all_accuracies.npy', all_accuracies)
np.save('/workspaces/MoBa_FP/Experiments/Attention_Exp_1/all_heads.npy', all_heads)

print("Training of all models completed!")