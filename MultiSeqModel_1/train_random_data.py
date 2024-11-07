from TransModel import MultiSeqModel
import tensorflow as tf
import numpy as np
from Files.attention_utils import create_combined_mask
import matplotlib.pyplot as plt
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


signals = []
seqs = []
data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"

num_sequences = 100 #7000 zum trainieren
num_reads = 10

# Find the maximum length across all signals for padding
max_length = 0
for i in range(num_sequences):
    for j in range(num_reads):
        signal = np.load(f"{data_path}/signal_seq_{i}_read_{j}.npy")
        max_length = max(max_length, signal.shape[0])
print(f"{max_length} is the longest length of a read")

# Load, pad, and store signals and sequences
for i in range(num_sequences):
    #List initialized
    sequence_signals = []
    for j in range(num_reads):
        # Load signal and pad to max_length
        signal = np.load(f"{data_path}/signal_seq_{i}_read_{j}.npy")
        padding_length = max_length - signal.shape[0]
        padded_signal = np.pad(signal, (0, padding_length), mode='constant', constant_values=0)
        sequence_signals.append(padded_signal)
        
    # Load target sequence
    signals.append(sequence_signals)
    seq = np.load(f"{data_path}/signal_seq_{i}_read_{0}_tarseq.npy")
    seqs.append(seq)

# Convert lists to arrays
signals = torch.from_numpy(np.array(signals))
seqs = torch.from_numpy(np.array(seqs))

signals = signals.view(signals.shape[0], signals.shape[1], signals.shape[2], 1).float()
print(signals.shape)
print(seqs.shape)
model = MultiSeqModel(max_length, 200)

# input_tensor_example = tf.convert_to_tensor(signal.reshape((1, len(signals[0]), 1)), dtype=tf.float32)
# signals = tf.convert_to_tensor(signals.reshape(signals.shape[0], signals.shape[1], 1), dtype=tf.float32)  # Form: (100, 2795, 1)
# seqs = tf.convert_to_tensor(seqs, dtype=tf.int32)  # Sequenzen als integer-Indizes für CTC Loss


model = MultiSeqModel(input_length=max_length, tar_length=200)
dataset = TensorDataset(signals, seqs)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
model.train_model(train_loader, num_epochs=20, learning_rate=0.001)
# Test forward pass with dummy input
# dummy_input = torch.randn(32, 10, input_length, 1)  # Batch of 32, 10 sequences, example length, 1 channel
# print(dummy_input.shape)

# output = model(dummy_input)
# print(output.shape)