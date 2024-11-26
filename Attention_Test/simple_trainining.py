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
from eval_utils import plot_training_curves_separate




# Train Paramaters
batch_size = 32
num_reads = 1
num_epochs = 150
lr_start = 0.00005
lr_end = 0.000003
dim_squeeze = True
train_seqs = 10000
test_seqs = 10000
plot_dir = f"/media/hdd1/MoritzBa/Plots/{train_seqs}_s_{num_epochs}_ep_{num_reads}_r.png"
output_dir_model = f"/media/hdd1/MoritzBa/Models/{train_seqs}_s_{num_epochs}_ep_{num_reads}_r.pth"
print(f"""
Training Process Details:
-------------------------
Batch Size: {batch_size}
Number of Reads: {num_reads}
Number of Epochs: {num_epochs}
Learning Rate Start: {lr_start}
Learning Rate End: {lr_end}
Dimensional Squeeze: {dim_squeeze}
Training Sequences: {train_seqs}
Testing Sequences: {test_seqs}
""")
#Prep
device = get_device(gpu_index=1)
data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"
max_length = 2100
max_length, train_loader = get_data_loader(data_path,train_seqs, batch_size = batch_size, num_reads=num_reads, dim_squeeze=True, overwrite_max_length = max_length)
max_2, test_loader = get_data_loader(data_path,end_sequence=train_seqs+test_seqs,start_sequence=train_seqs, batch_size = batch_size, num_reads=num_reads, dim_squeeze= True, overwrite_max_length= max_length)


#Create Model and Train
model = BasicAtt(input_length=max_length, tar_length=200,d_model = 64, max_pool_id = 1, multi_seq_nr=num_reads)
losses,n_accuracies, ham_accuracies,test_accs = model.train_model(train_loader, num_epochs=num_epochs, learning_rate=lr_start,
                                                                   device=device, test_set=test_loader, save_path=output_dir_model, lr_end = lr_end)
    

criterion = nn.CrossEntropyLoss()
train_loss, train_accuracy, train_ham_ac = evaluate_model(model, train_loader, criterion, device)
print(f"Final Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Training Lev-Accuracy: {train_ham_ac:.2f}%")
    
# Evaluate model on test data
test_loss, test_accuracy,test_ham_ac = evaluate_model(model, test_loader, criterion, device)
print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%,  Test Lev-Accuracy: {test_ham_ac:.2f}%")
    
plot_training_curves_separate(losses,n_accuracies, ham_accuracies,test_accs,plot_dir)

