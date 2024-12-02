from TransCTCMultiModel import MultiCTC
from CTC_Test import CTC_Test_Model
import numpy as np
import matplotlib.pyplot as plt
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_prep_func import get_data_loader
from data_prep_func import get_device
from data_prep_func import vectors_to_sequence
from eval_utils import evaluate_model_ham
from eval_utils import plot_training_curves_separate




# Train Paramaters
batch_size = 16
num_reads = 1
learning_rate = 0.001
n_heads = 16
at_dim = 16
at_layer = 0
dim_squeeze = True
train_seqs = 150000
test_seqs = 2500
num_epochs = 50
plot_dir = f"/media/hdd1/MoritzBa/Plots/CTC_{train_seqs}_s_{num_epochs}_ep_{num_reads}_r.png"
output_dir_model = f"/media/hdd1/MoritzBa/Models/CTC_{train_seqs}_s_{num_epochs}_ep_{num_reads}_r.pth"
print(f"""
Training Process Details of Multi CTC Training:
-------------------------
Batch Size: {batch_size}
Heads : {n_heads}
At Dim : {at_dim}
Attention Layers : {at_layer}
Number of Reads: {num_reads}
Number of Epochs: {num_epochs}
Start Learning Rate: {learning_rate}
Dimensional Squeeze: {dim_squeeze}
Training Sequences: {train_seqs}
Testing Sequences: {test_seqs}
""")
#Prep
device = get_device(gpu_index=0)
data_path = "/media/hdd1/MoritzBa/Big_Data/Rd_Data_Numpy"
max_length = 2100
max_length, train_loader = get_data_loader(data_path,train_seqs, batch_size = batch_size, num_reads=num_reads, dim_squeeze=True, overwrite_max_length = max_length)
max_2, test_loader = get_data_loader(data_path,end_sequence=train_seqs+test_seqs,start_sequence=train_seqs, batch_size = batch_size, num_reads=num_reads, dim_squeeze= True, overwrite_max_length= max_length)


#Create Model and Train
model = CTC_Test_Model(input_length=max_length, tar_length=200, conv_1_dim=1,conv_2_dim=1, attention_dim=at_dim, num_reads=num_reads,
                 n_heads = n_heads, at_layer = at_layer)
losses,n_accuracies, ham_accuracies,test_accs = model.train_model(train_loader, num_epochs=num_epochs, learning_rate=learning_rate,
                                                                   device=device, test_set=test_loader, save_path=output_dir_model,scheduler_type="cosine_restart")
    


train_ham_ac = evaluate_model_ham(model, train_loader,device)
print(f" Training Lev-Accuracy: {train_ham_ac:.2f}%")
    
# Evaluate model on test data
test_ham_ac = evaluate_model_ham(model, test_loader, device)
print(f"Test Lev-Accuracy: {test_ham_ac:.2f}%")
    
plot_training_curves_separate(losses,n_accuracies, ham_accuracies,test_accs,plot_dir)

