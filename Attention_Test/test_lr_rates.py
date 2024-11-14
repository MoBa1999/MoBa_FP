import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Simple_Multi import BasicMulti
from data_prep_func import get_data_loader, get_device
from eval_utils import evaluate_model

# Set up device and data paths
device = get_device(gpu_index=0)
data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"

# Define range of learning rates
learning_rates = np.logspace(-5, -4, 5)  # 20 values between 0.0001 and 0.1

# Initialize storage for results
train_losses = []
train_accuracies = []
test_accuracies = []

# Loop over learning rates
for lr in learning_rates:
    # Prepare data loaders
    max_length, train_loader = get_data_loader(data_path, 1000, batch_size=32, num_reads=1)
    
    # Initialize model
    model = BasicMulti(input_length=max_length, tar_length=200, d_model=64, max_pool_id=1).to(device)
    
    # Train model and record losses and accuracies
    losses, accuracies = model.train_model(train_loader, num_epochs=100, learning_rate=lr, device=device)
    train_losses.append(losses)
    train_accuracies.append(accuracies)
    
    # Test model on evaluation data
    criterion = nn.CrossEntropyLoss()
    max_length, test_loader = get_data_loader(data_path, end_sequence=1500, start_sequence=1000, batch_size=32, num_reads=1)
    _, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    test_accuracies.append(test_accuracy)

    print(f"Learning Rate: {lr:.4f}, Final Train Loss: {losses[-1]:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Plot training loss curves for each learning rate
plt.figure(figsize=(12, 6))
for i, lr in enumerate(learning_rates):
    plt.plot(train_losses[i], label=f"LR={lr:.5f}")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Loss Curves for Different Learning Rates")
plt.legend(loc="upper right")
plt.show()

# Plot training accuracy curves for each learning rate
plt.figure(figsize=(12, 6))
for i, lr in enumerate(learning_rates):
    plt.plot(train_accuracies[i], label=f"LR={lr:.5f}")
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy Curves for Different Learning Rates")
plt.legend(loc="lower right")
plt.show()

# Plot test accuracy over learning rate
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, test_accuracies, marker="o")
plt.xscale("log")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs. Learning Rate")
plt.show()
