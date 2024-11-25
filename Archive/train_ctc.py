from TransCTCModel import MultiSeqCTCModel
import numpy as np
import matplotlib.pyplot as plt
import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_prep_func import get_data_loader
from data_prep_func import get_device
from data_prep_func import vectors_to_sequence
from data_prep_func import decode_ctc_output
from data_prep_func import collapse_string_ctc
from eval_utils import evaluate_model_ham
from Levenshtein import distance


device = get_device(gpu_index=2)


data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"
max_length, train_loader = get_data_loader(data_path,40000, batch_size = 16, dim_squeeze=True, num_reads=1)
max_length_1, test_loader = get_data_loader(data_path,start_sequence=40000, end_sequence=45000, batch_size = 16, dim_squeeze=True, num_reads=1)
max_length = max(max_length,max_length_1)



training_accuracies = []
test_accuracies = []
output_dir = "/workspaces/MoBa_FP/Experiments/....!!!!!!!!!!!!!"
seqs = [1000, 5000, 10000, 20000, 40000]

for seq in seqs:
    _, train_loader = get_data_loader(data_path,seq, batch_size = 16, dim_squeeze=True, num_reads=1)
    model = MultiSeqCTCModel(input_length=max_length, tar_length=200, conv_1_dim=16,conv_2_dim=48, attention_dim=64)
    # Train model and get losses and accuracies
    losses, hammings, accuracies = model.train_model(train_loader, num_epochs=800, learning_rate=0.001, device=device, scheduler_type="cosine_restart")

    train_accuracy = evaluate_model_ham(model, train_loader, device)
    print(f"Training Accuracy: {train_accuracy:.2f}%")

    test_accuracy = evaluate_model_ham(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    training_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)



np.save(os.path.join(output_dir, "training_accuracies.npy"), training_accuracies)
np.save(os.path.join(output_dir, "test_accuracies.npy"), test_accuracies)
np.save(os.path.join(output_dir, "end_seqs.npy"), seqs)


# plt.plot(losses)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss")
# plt.show()

# plt.figure
# plt.plot(hammings)
# plt.xlabel("Epoch")
# plt.ylabel("Hamming Distance Average")
# plt.title("Training Values")
# plt.show()

# plt.figure
# plt.plot(accuracies)
# plt.xlabel("Epoch")
# plt.ylabel("Theoretical Accuracy")
# plt.title("Training Values")
# plt.show()




