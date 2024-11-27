import matplotlib.pyplot as plt

# Daten
training_sequences = [4000, 6000, 8000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000]
test_lev_accuracy = [50.58, 50.74, 50.7725, 50.973, 50.95, 51.455, 51.6445, 52.3215, 52.754, 52.912, 53.296, 53.627]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(training_sequences, test_lev_accuracy, marker='o', linestyle='-', color='b', label="Lev Accuracy")

# Achsenbeschriftung und Titel
plt.xlabel("Training Sequences")
plt.ylabel("Test Lev Accuracy (%)")
plt.title("Training Sequences vs Test Lev Accuracy")
plt.ylim(40,80)
plt.xlim(0,250000)
plt.grid(True)
plt.legend()

# Plot anzeigen
plt.show()