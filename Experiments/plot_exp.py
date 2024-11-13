import numpy as np
import matplotlib.pyplot as plt

# Lade die gespeicherten Daten
all_losses = np.load('/workspaces/MoBa_FP/Experiments/Attention_Exp_0/all_losses.npy')
all_accuracies = np.load('/workspaces/MoBa_FP/Experiments/Attention_Exp_0/all_accuracies.npy')
all_attention_dims = np.load('/workspaces/MoBa_FP/Experiments/Attention_Exp_0/all_attention_dims.npy')

# Extrahiere eindeutige Aufmerksamkeitdimensionen
unique_dims = np.unique(all_attention_dims)

# Initialisiere Listen, um den letzten Verlust und die letzte Genauigkeit für jede Dimension zu speichern
last_losses = []
last_accuracies = []

for dim in unique_dims:
  # Filtere Daten für die aktuelle Aufmerksamkeitdimension
  filtered_losses = all_losses[all_attention_dims == dim]
  filtered_accuracies = all_accuracies[all_attention_dims == dim]

  # Nimm den letzten Verlust und die letzte Genauigkeit
  last_loss = filtered_losses[0,-1]
  last_accuracy = filtered_accuracies[0,-1]

  # Füge zu den entsprechenden Listen hinzu
  last_losses.append(last_loss)
  last_accuracies.append(last_accuracy)

# Plotte die Ergebnisse
plt.figure(figsize=(12, 6))

# Plotte den letzten Verlust
plt.subplot(1, 2, 1)
plt.plot(unique_dims, last_losses, label='Letzter Verlust')
plt.xlabel('Attentiondimension')
plt.ylabel('Loss')
plt.title('Loss')
plt.grid(True)

# Plotte die letzte Genauigkeit
plt.subplot(1, 2, 2)
plt.plot(unique_dims, last_accuracies, label='Letzte Genauigkeit', color='orange')
plt.xlabel('Attentiondimension')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Daten erfolgreich geladen und geplottet!")

# Extrahiere eindeutige Aufmerksamkeitdimensionen
unique_dims = np.unique(all_attention_dims)
plt.figure()
# Iteriere über jede Aufmerksamkeitdimension und erstelle eine separate Figur
for dim in unique_dims:
    # Filtere Daten für die aktuelle Aufmerksamkeitdimension
    filtered_losses = all_losses[all_attention_dims == dim]

    # Erstelle eine neue Figur
 
    plt.plot(filtered_losses[0,:])
    plt.xlabel('Epoche')
    plt.ylabel('Verlust')
    plt.title(f'Verlustkurve für Aufmerksamkeitdimension {dim}')
    plt.grid(True)
plt.show()

print("Daten erfolgreich geladen und geplottet!")