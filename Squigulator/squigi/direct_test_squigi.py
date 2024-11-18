import subprocess
import h5py
import pyslow5
import os
import numpy as np
import random 
import string
import re
import matplotlib.pyplot as plt




def save_fasta(sequence, output_file, header="Generated_Sequence"):
    """
    Saves a given sequence to a FASTA file.

    Args:
        sequence: A string containing the sequence (e.g., "ATGC...").
        output_file: The path to the FASTA file to save.
        header: The header for the FASTA file (default: "Generated_Sequence").
    """
    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the sequence to the FASTA file
    with open(output_file, 'w') as f:
        f.write(f">{header}\n")
        f.write(sequence + "\n")

def run_squigulator(fasta_folder, blow5_folder, reads_per_sequence):
    """
    Runs the squigulator command for multiple training files.

    Parameters:
        num_files (int): The number of files to process (default: 100)
    """
    
    input_file = fasta_folder
    for j in range (reads_per_sequence):
        output_file = blow5_dir
        print(output_file)
            # Command to be executed
        command = ["./squigulator", "-x", "dna-r9-min", input_file, "-o", output_file, "-n", "1", "--ideal"]
        
            # Run the command
        subprocess.run(command, check=True)


def get_blow5_data(blow_file):
    print("File will be read")
    try:
    # Open the blow5 file and read data
        s5 = pyslow5.Open(blow_file, 'r')
        reads = s5.seq_reads()
        print(reads)
        for read in reads:
            read_id = read['read_id']
            signal = read['signal']
            print(signal)
                        
            # Convert signal to NumPy array
            signal_array = np.array(signal, dtype=np.int16) 
            print("done")               
    except:
        print(f"BLOW5 file not found: {blow_file}")
    print("hi")       
    return signal_array

output_dir = "/workspaces/MoBa_FP/Squigulator/squigi/temp/temp_1.fasta"
blow5_dir = "/workspaces/MoBa_FP/Squigulator/squigi/temp/temp_1.blow5"
seq = "AAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAATAAAAAAAAAT"
save_fasta(seq,output_dir)
run_squigulator(output_dir,blow5_dir,1)
signal = get_blow5_data(blow5_dir)
plt.plot(signal[0:200])
plt.xlabel(seq[0:40])
plt.ylabel("Current Value")
plt.show()
