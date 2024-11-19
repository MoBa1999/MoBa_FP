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
max_length, train_loader = get_data_loader(data_path,500, batch_size = 16, dim_squeeze=True, num_reads=1)

model = MultiSeqCTCModel(input_length=max_length, tar_length=200,  conv_1_dim=16,conv_2_dim=48, attention_dim=64)


# 
    # Train model and get losses and accuracies
losses, accuracies = model.train_model(train_loader, num_epochs=100, learning_rate=0.000005, device=device)
data, target = next(iter(train_loader))
seq = vectors_to_sequence(target[0].numpy())
print(f"Soll Sequenz: {seq}")
first = data[0]
first = first.to(device)
output = model(first.unsqueeze(0))
output = output.cpu().detach().numpy()
output = output.squeeze(0)
out = decode_ctc_output(output)
print(f"Output from CTC: {out}")
losses, accuracies = model.train_model(train_loader, num_epochs=100, learning_rate=0.000005, device=device)
data, target = next(iter(train_loader))
seq = vectors_to_sequence(target[0].numpy())
print(f"Soll Sequenz: {seq}")
first = data[0]
first = first.to(device)
output = model(first.unsqueeze(0))
output = output.cpu().detach().numpy()
output = output.squeeze(0)
out = decode_ctc_output(output)
print(f"Output from CTC: {out}")
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