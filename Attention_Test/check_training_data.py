import numpy as np
import matplotlib.pyplot as plt
import os 
import subprocess
import pyslow5


def plot_squigulator(fasta_folder, blow5_folder, reads_per_sequence):

    
    input_file = f"{fasta_folder}/fasta_file_{1000}.fasta"
    for j in range (reads_per_sequence):
        output_file = f"{blow5_folder}/seq_{2}_read_{j}.blow5"
        # Command to be executed
        if j != 0:
            command = ["/workspaces/MoBa_FP/Squigulator/squigi/squigulator", "-x", "dna-r9-min", input_file, "-o", output_file, "-n", "1", "--seed", str(j+1)]
        else:
            command = ["/workspaces/MoBa_FP/Squigulator/squigi/squigulator", "-x", "dna-r9-min", input_file, "-o", output_file, "-n", "1", "--ideal"]
        

        # Run the command
        subprocess.run(command, check=True)
        s5 = pyslow5.Open(output_file, 'r')
        reads = s5.seq_reads()
        for read in reads:
            signal = read['signal']
            plt.plot(signal)

fasta_dir = "/media/hdd1/MoritzBa/Rd_Data_Fasta"
example_folder = "/workspaces/MoBa_FP/Training_Data_Example"
plot_squigulator(fasta_dir,example_folder,1)

signals = []
seqs = []
data_path = "/media/hdd1/MoritzBa/Rd_Data_Numpy"

sequence = 1000 #7000 zum trainieren
num_reads = [0]
for j in num_reads:
    # Load signal and pad to max_length
    signal = np.load(f"{data_path}/signal_seq_{sequence}_read_{j}.npy")
    print(signal)
    plt.plot(signal)



plt.xlim(0,400)
plt.show()
        

