from TransModel import MultiSeqModel
import numpy as np
from Files.attention_utils import create_combined_mask
import matplotlib.pyplot as plt
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_prep_func import get_data_loader



data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"

max_length, train_loader = get_data_loader(data_path,1000)

model = MultiSeqModel(max_length, 200)

# input_tensor_example = tf.convert_to_tensor(signal.reshape((1, len(signals[0]), 1)), dtype=tf.float32)
# signals = tf.convert_to_tensor(signals.reshape(signals.shape[0], signals.shape[1], 1), dtype=tf.float32)  # Form: (100, 2795, 1)
# seqs = tf.convert_to_tensor(seqs, dtype=tf.int32)  # Sequenzen als integer-Indizes für CTC Loss


model = MultiSeqModel(input_length=max_length, tar_length=200)
print(model.get_num_params())

model.train_model(train_loader, num_epochs=100, learning_rate=0.001)


model.eval()  # Set model to evaluation mode
correct_predictions = 0
total_predictions = 0

with torch.no_grad():  # Disable gradient calculation for evaluation
    for inputs, labels in train_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=-1)  # Predicted classes
        correct_predictions += (predicted == labels.argmax(dim=-1)).sum().item()
        total_predictions += labels.numel()

overall_accuracy = 100 * correct_predictions / total_predictions
print(f"Overall Training Accuracy: {overall_accuracy:.2f}%")