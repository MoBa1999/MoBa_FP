from Simple_Model import BasicModel
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
max_length, train_loader = get_data_loader(data_path,10, batch_size = 2, num_reads=1)


model = BasicModel(input_length=max_length, tar_length=200,d_model = 64, cnn_blocks = 3, max_pool_id = 1)
    
    # Train model and get losses and accuracies
losses, accuracies = model.train_model(train_loader, num_epochs=100, learning_rate=0.1, device=device)

