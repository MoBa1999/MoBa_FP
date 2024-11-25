from torch.utils.data import DataLoader, TensorDataset
from data_prep_func import get_data_loader
from data_prep_func import get_device
from data_prep_func import vectors_to_sequence
from eval_utils import evaluate_model
from eval_utils import plot_training_curves_separate
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from Simple_Attention import BasicAtt

test_path = "/media/hdd1/MoritzBa/Test_Data/Rd_Data_Numpy"
model_path_general = "/media/hdd1/MoritzBa/Models"

max_length, test_loader = get_data_loader(test_path, 20000, batch_size=32, num_reads=1)

models = [
    "10000_s_75_ep.pth", "10000_s_75_ep_5_r.pth", "20000_s_75_ep_2_r.pth", "40000_s_75_ep_10_r.pth",
    "10000_s_75_ep_10_r.pth", "20000_s_75_ep.pth", "20000_s_75_ep_5_r.pth", "40000_s_75_ep_2_r.pth",
    "10000_s_75_ep_2_r.pth", "20000_s_75_ep_10_r.pth", "40000_s_75_ep.pth", "40000_s_75_ep_5_r.pth"
]
labels = ['10000 Seqs - 1 R/S', '10000 Seqs - 5 R/S', '20000 Seqs - 2 R/S', '40000 Seqs - 10 R/S',
          '10000 Seqs - 10 R/S', '20000 Seqs - 1 R/S', '20000 Seqs - 5 R/S', '40000 Seqs - 2 R/S',
          '10000 Seqs - 2 R/S', '20000 Seqs - 10 R/S', '40000 Seqs - 75 R/S', '40000 Seqs - 5 R/S']
nr = [1,5,2,10,10,1,5,2,2,10,1,5]
seqs = [10000, 10000, 20000, 40000, 10000, 20000, 20000, 40000, 10000, 20000, 40000, 40000]
device = get_device(gpu_index=1)

test_lev_accuracies = []

for model_name, rs in zip(models,nr):
    model_path = os.path.join(model_path_general, model_name)
    print(f"The following model is analyzed: {model_path}" )
    model = BasicAtt(input_length=max_length, tar_length=200,d_model = 64, max_pool_id = 1, multi_seq_nr=rs)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_lev_ac = evaluate_model(model, test_loader, criterion, device)

    test_lev_accuracies.append(test_lev_ac)

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(seqs, test_lev_accuracies)

# Add labels to each point
for i, (seq, lev_ac, label) in enumerate(zip(seqs, test_lev_accuracies, labels)):
    plt.annotate(label, (seq, lev_ac), textcoords="offset points", xytext=(10, 10), ha='center')

plt.xlabel('Sequence Length')
plt.ylabel('Test Levenshtein Accuracy')
plt.title('Test Levenshtein Accuracy vs. Sequence Length')
plt.grid(True)
plt.show()