from Simple_Multi import BasicMulti
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
output_dir = "/workspaces/MoBa_2/MoBa_FP/Experiments/Exp4"
max_length, train_loader = get_data_loader(data_path,1000, batch_size = 32, num_reads=2, dim_squeeze=True)

training_losses = []
training_accuracies = []
test_losses = []
test_accuracies = []
inputs = list(range(1, 11))

for input in inputs:
    max_length, train_loader = get_data_loader(data_path,1000, batch_size = 32, num_reads=input, dim_squeeze=True)

    model = BasicMulti(input_length=max_length, tar_length=200,d_model = 64, max_pool_id = 1, multi_seq_nr=input)
    
    # Train model and get losses and accuracies
    losses, accuracies = model.train_model(train_loader, num_epochs=200, learning_rate=0.0005, device=device)

    criterion = nn.CrossEntropyLoss()  # Define loss function for evaluation
    eval_loss, eval_accuracy = evaluate_model(model, train_loader, criterion, device)
    training_losses.append(eval_loss)
    training_accuracies.append(eval_accuracy)

    # Print evaluation results
    print(f"Training Loss: {eval_loss:.4f}, Training Accuracy: {eval_accuracy:.2f}%")

    max_length, test_loader = get_data_loader(data_path,end_sequence=1500,start_sequence=1000, batch_size = 32, num_reads=input, dim_squeeze= True)
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {eval_loss:.4f}, Test Accuracy: {eval_accuracy:.2f}%")
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

np.save(os.path.join(output_dir, "training_losses.npy"), np.array(training_losses))
np.save(os.path.join(output_dir, "training_accuracies.npy"), np.array(training_accuracies))
np.save(os.path.join(output_dir, "test_losses.npy"), np.array(test_losses))
np.save(os.path.join(output_dir, "test_accuracies.npy"), np.array(test_accuracies))
np.save(os.path.join(output_dir, "seq_in.npy"), np.array(inputs))
