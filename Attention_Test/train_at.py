import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Simple_Attention import BasicAtt
from data_prep_func import get_data_loader, get_device
from eval_utils import evaluate_model

# Set up device and data paths
device = get_device(gpu_index=0)
data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"


# Initialize storage for results
train_losses = []
train_accuracies = []
test_accuracies = []


    # Prepare data loaders
max_length, train_loader = get_data_loader(data_path, 1000, batch_size=32, num_reads=10,dim_squeeze=True)
    
    # Initialize model
model = BasicAtt(input_length=max_length, tar_length=200, d_model=64, max_pool_id=1, multi_seq_nr = 10).to(device)
    
    # Train model and record losses and accuracies
losses, accuracies = model.train_model(train_loader, num_epochs=50, learning_rate=0.00006, device=device)

    
    # Test model on evaluation data
criterion = nn.CrossEntropyLoss()
max_length, test_loader = get_data_loader(data_path, end_sequence=1500, start_sequence=1000, batch_size=32, num_reads=10,dim_squeeze=True)
_, test_accuracy = evaluate_model(model, test_loader, criterion, device)


print(f"Final Train Loss: {losses[-1]:.4f}, Test Accuracy: {test_accuracy:.2f}%")


