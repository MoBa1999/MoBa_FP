from TransCTCModel import MultiSeqCTCModel
import numpy as np
import matplotlib.pyplot as plt
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_prep_func import get_data_loader
from data_prep_func import get_device


device = get_device(gpu_index=2)
device = None

data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"
max_length, train_loader = get_data_loader(data_path,100, batch_size = 4, dim_squeeze=False, num_reads=1)


model = MultiSeqCTCModel(input_length=max_length, tar_length=200, conv_1_dim=5, attention_dim=20)
    
    # Train model and get losses and accuracies
losses, accuracies = model.train_model(train_loader, num_epochs=100, learning_rate=0.00005, device=None)

