import numpy as np


signals = []
seqs = []
data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"

sequence = 1 #7000 zum trainieren
num_reads = 10

print("Start")

for j in range(num_reads):
    # Load signal and pad to max_length
    signal = np.load(f"{data_path}/signal_seq_{sequence}_read_{j}.npy")
    print(signal)
        

