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
from CTC_Test import CTC_Test_Model

#test_path = "/media/hdd1/MoritzBa/Test_Data/Rd_Data_Numpy"
test_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"

model_path_general = "/media/hdd1/MoritzBa/Models"


test_lev_accuracies = {}

seqs = [70000]
nr = [10]
device = get_device(gpu_index=1)
max_length = 2100

#CTC_70000_s_200_ep_10_r

# Initialisiere die Struktur
for r in nr:
    test_lev_accuracies[r] = []

for seq in seqs:
    for r in nr:
        model_name = f"CTC_{seq}_s_200_ep_{r}_r.pth"
        model_path = os.path.join(model_path_general, model_name)
        print(f"The following model is analyzed: {model_path}")
        
        _, test_loader = get_data_loader(test_path, end_sequence=42000,start_sequence=40000, batch_size=32, num_reads=r, overwrite_max_length=max_length, dim_squeeze=True)
        
        model = CTC_Test_Model(input_length=max_length, tar_length=200, d_model=64, max_pool_id=1, multi_seq_nr=r)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        model.to(device)
        criterion = nn.CrossEntropyLoss()
        test_loss, test_accuracy, test_lev_ac = evaluate_model(model, test_loader, criterion, device)

        # Speichere die Levenshtein-Test-Genauigkeiten in der Struktur
        test_lev_accuracies[r].append(test_lev_ac)





    

# Create the plot
plt.figure(figsize=(10, 6))


for r in nr:
    plt.scatter(seqs, test_lev_accuracies[r], label = f"")
        

plt.xlabel('Sequence Length')
plt.ylabel('Test Levenshtein Accuracy')
plt.title('Test Levenshtein Accuracy vs. Sequence Length')
plt.legend()
plt.grid(True)
plt.show()