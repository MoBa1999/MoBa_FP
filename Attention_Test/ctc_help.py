
import tensorflow as tf
import numpy as np
from Files.attention_utils import create_combined_mask
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os
import torch.nn.functional as F

device = torch.device('cpu')
# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def collapse_sequences(sequences):
    collapsed = []
    for seq in sequences:
        # Filter out zeros and collapse consecutive duplicates
        filtered_seq = [x for x in seq if x != 0]
        collapsed_seq = [filtered_seq[0]] if filtered_seq else []
        
        for i in range(1, len(filtered_seq)):
            if filtered_seq[i] != filtered_seq[i - 1]:
                collapsed_seq.append(filtered_seq[i])
        
        collapsed.append(collapsed_seq)

    # Determine the max length for padding
    max_length = max(len(seq) for seq in collapsed)

    # Pad sequences to the right with zeros
    padded = np.array([seq + [0] * (max_length - len(seq)) for seq in collapsed])
    
    return padded

with tf.device('/CPU:0'):
    # Custom CTC loss function
    Test_Object = tf.keras.losses.CTC(name="Customized")
    print(Test_Object.get_config())

    #Pytorch Variant
    ctc_loss_torch = nn.CTCLoss()

    # Define parameters
    batch_size = 2
    max_length = 10
    num_classes = 2

    # Create y_true (integer labels) with values in range [0, num_classes - 1]
    #y_true = np.random.randint(1, num_classes, size=(batch_size, max_length), dtype=int)
    y_true = np.array([[1, 2,1]])

    print("y_true:", y_true)

    # Create y_pred (logits) with high values in places that match y_true
    #y_pred = np.full((batch_size, max_length, num_classes), -100.0)
    y_pred  = np.array([[
        [100, -100, -100], #0
        [-100, 100, -100], #1
        [-100, -100, 100], #2
        [-100, 100, -100], #1
        [-100, 100, -100] #1
    ]])
    # y_pred = np.random.uniform(low=-100, high=100, size=(2, 5, 3))
    # print("y_pred:", y_pred)

    # y_true_c = collapse_sequences(np.argmax(y_pred, axis=2))
    # print(y_true_c)

    #print(f"calculated y_true: {y_true}")
    print(f"Real y_true: {y_true}")
    # Calculate CTC Loss
    loss = Test_Object(y_true, y_pred)
    print(f"DER CTC LOSS: {loss}")
    # Convert to PyTorch tensors

#Change to torch
y_true_torch = torch.tensor(y_true, dtype=torch.long)  # Shape (batch_size, target_length)
y_pred_torch = torch.tensor(y_pred, dtype=torch.float)

# Create input_lengths and target_lengths
input_lengths = torch.tensor([y_pred_torch.size(0)], dtype=torch.long)  # Length of each sequence in y_pred
#y_true_torch = torch.randint(1, 20, (1, 3), dtype=torch.long)
target_lengths = torch.tensor([len(seq) for seq in y_true], dtype=torch.long)  # Length of each target sequence

    #loss = ctc_loss_torch(y_pred_torch, y_pred_torch, input_lengths, target_lengths)
ctc_loss = nn.CTCLoss()
# Example input tensors
#T = 30, N = 10, C = 29

y_pred_2  = np.array([[
        [100, -100, -100]], #0
        [[-100, 100, -100]], #1
        [[-100, 100, -100]], #1
        [[-100, -100, 100]], #2
        [[100, -100, -100]] #0
    ])

targ = np.array([[1, 2]])
#log_probs = torch.randn(4, 1, 3)  # (T, N, C)
log_probs = torch.from_numpy(y_pred_2).float()
log_probs = F.log_softmax(log_probs)
#log_probs = torch.log(log_probs)
print(log_probs.shape)
#targets = torch.randint(0, 2, (1, 2), dtype=torch.long)  # (N, S)
targets = torch.from_numpy(targ).float()
input_lengths = torch.tensor([5] * 1)  # (N)
target_lengths = torch.tensor([2] * 1)  # (N)

print(log_probs.numpy())
print(targets.numpy())


# Calculate CTC loss
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
print(loss)


