import os
import numpy as np
from Simple_Model import BasicModel
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
from eval_utils import plot_training_curves

device = get_device(gpu_index=1)
# Beispielpfad zum Speichern der Ergebnisse
output_dir = "/workspaces/MoBa_2/MoBa_FP/Experiments/Exp2"
os.makedirs(output_dir, exist_ok=True)  # Ordner erstellen, falls nicht vorhanden

# Daten laden und in Trainings- und Testdaten aufteilen
data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"
max_length_total, full_data = get_data_loader(data_path, end_sequence=10000, batch_size=32,num_reads=1)
_, test_data = get_data_loader(data_path,start_sequence=8000, end_sequence=10000, batch_size=32, overwrite_max_length=max_length_total,num_reads=1)
# Für jedes end_sequence die Modelle trainieren und die Ergebnisse speichern
training_losses = []
training_accuracies = []
test_losses = []
test_accuracies = []

end_seqs = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
criterion = nn.CrossEntropyLoss()

for s in end_seqs:

    max_length, train_data = get_data_loader(data_path, end_sequence=s, batch_size=32, overwrite_max_length=max_length_total,num_reads=1)
    # Modell instanziieren und trainieren
    model = BasicModel(input_length=max_length_total, tar_length=200, d_model=64, max_pool_id=1)
    losses, accuracies = model.train_model(train_data, num_epochs=200, learning_rate=0.0005, device=device)

    # Trainingsergebnisse speichern
    eval_loss, eval_accuracy = evaluate_model(model, train_data, criterion, device)
    training_losses.append(eval_loss)
    training_accuracies.append(eval_accuracy)

    # Testen auf dem TestSet und die Ergebnisse speichern
    test_loss, test_accuracy = evaluate_model(model, test_data, criterion, device)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

# Ergebnisse als numpy-Arrays speichern
np.save(os.path.join(output_dir, "training_losses.npy"), np.array(training_losses))
np.save(os.path.join(output_dir, "training_accuracies.npy"), np.array(training_accuracies))
np.save(os.path.join(output_dir, "test_losses.npy"), np.array(test_losses))
np.save(os.path.join(output_dir, "test_accuracies.npy"), np.array(test_accuracies))
np.save(os.path.join(output_dir, "end_seqs.npy"), np.array(end_seqs))
