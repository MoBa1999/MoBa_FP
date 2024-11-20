from Archive.Simple_Model import BasicModel
import numpy as np
import matplotlib.pyplot as plt
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_prep_func import get_data_loader
from data_prep_func import get_device
from data_prep_func import vectors_to_sequence
from eval_utils import evaluate_model
from eval_utils import plot_training_curves

device = get_device(gpu_index=1)

data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"
max_length, train_loader = get_data_loader(data_path,1000, batch_size = 32, num_reads=1)


model = BasicModel(input_length=max_length, tar_length=200,d_model = 64, cnn_blocks = 3, max_pool_id = 1)
    
    # Train model and get losses and accuracies
losses, accuracies = model.train_model(train_loader, num_epochs=200, learning_rate=0.0005, device=device)

criterion = nn.CrossEntropyLoss()  # Define loss function for evaluation
eval_loss, eval_accuracy = evaluate_model(model, train_loader, criterion, device)

# Print evaluation results
print(f"Training Loss: {eval_loss:.4f}, Training Accuracy: {eval_accuracy:.2f}%")

max_length, train_loader = get_data_loader(data_path,end_sequence=1500,start_sequence=1000, batch_size = 32, num_reads=1)
eval_loss, eval_accuracy = evaluate_model(model, train_loader, criterion, device)

print(f"Test Loss: {eval_loss:.4f}, Test Accuracy: {eval_accuracy:.2f}%")
