from scipy.stats import pearsonr

import numpy as np
import matplotlib.pyplot as plt
def calculate_average_correlation(sequences):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = [np.pad(seq, (0, max_length - len(seq)), constant_values=0) for seq in sequences]
    padded_sequences = np.array(padded_sequences)
    sequences = padded_sequences
    n = len(sequences)
    total_correlation = 0
    num_pairs = 0
    corrs_ = []

    # Berechne die paarweise Korrelation
    for i in range(n):
        for j in range(i + 1, n):
            # Berechne die Pearson-Korrelation für jedes Paar von Sequenzen
            corr, _ = pearsonr(sequences[i], sequences[j])
            total_correlation += corr
            num_pairs += 1
            corrs_.append(corr)

    # Durchschnittliche Korrelation
    return corrs_




seqs = []
data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"

sequences = 5000
correlations = []
for i in range(sequences):
    num_reads = [1,2,3,4,5,6,7,8,9]
    signals = []
    for j in num_reads:
        # Load signal and pad to max_length
        signal = np.load(f"{data_path}/signal_seq_{i}_read_{j}.npy")
        signals.append(signal)
        correlations.extend(calculate_average_correlation(signals))

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(correlations, bins=30, edgecolor='black', alpha=0.7)  # Adjust the number of bins as needed
plt.title('Histogram of Correlations Values')
plt.xlabel('Correlation Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()