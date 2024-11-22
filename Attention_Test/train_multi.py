from Simple_Attention import BasicAtt
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

device = get_device(gpu_index=2)
data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"
max_length, train_loader = get_data_loader(data_path,45, batch_size = 32, num_reads=1, dim_squeeze=True)

_, test_loader = get_data_loader(data_path,end_sequence=45000,start_sequence=40000, batch_size = 32, num_reads=1, dim_squeeze= True)
    # Train model and get losses and accuracies

# Initialize lists to store results
training_losses = []
training_accuracies = []
test_losses = []
test_accuracies = []
output_dir = "/workspaces/MoBa_FP/Experiments/Exp_3_at"
batch_size = 32
num_reads = 1
num_epochs = 75
learning_rate = 0.00007
dim_squeeze = True
end_sequences = [1000, 5000, 10000, 20000, 40000]

# Loop through different end_sequence values
for end_sequence in end_sequences:
    print(f"Training with end_sequence = {end_sequence}...")
    
    # Load data
    model = BasicAtt(input_length=max_length, tar_length=200,d_model = 64, max_pool_id = 1, multi_seq_nr=num_reads)
    _, train_loader = get_data_loader(data_path, end_sequence=end_sequence, batch_size=batch_size, num_reads=num_reads, dim_squeeze=dim_squeeze)
    
    # Train model
    losses,n_accuracies, ham_accuracies = model.train_model(train_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)
    
    # Evaluate model on training data
    criterion = nn.CrossEntropyLoss()
    train_loss, train_accuracy, train_ham_ac = evaluate_model(model, train_loader, criterion, device)
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Training Ham-Accuracy: {train_ham_ac:.2f}%")
    
    # Evaluate model on test data
    test_loss, test_accuracy,test_ham_ac = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%,  Test Ham-Accuracy: {test_ham_ac:.2f}%")
    
    # Store results
    training_losses.append(train_loss)
    training_accuracies.append(train_ham_ac)
    test_losses.append(test_loss)
    test_accuracies.append(test_ham_ac)

# Save results to .npy files
np.save(os.path.join(output_dir, "training_losses.npy"), training_losses)
np.save(os.path.join(output_dir, "training_accuracies.npy"), training_accuracies)
np.save(os.path.join(output_dir, "test_losses.npy"), test_losses)
np.save(os.path.join(output_dir, "test_accuracies.npy"), test_accuracies)
np.save(os.path.join(output_dir, "end_seqs.npy"), end_sequences)

torch.save(model.state_dict(), '/media/hdd1/MoritzBa/Models/3_in_at.pth')

print("Model saved.")
