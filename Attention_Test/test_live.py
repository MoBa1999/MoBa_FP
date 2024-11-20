import numpy as np
import matplotlib.pyplot as plt
import os 
import subprocess
import pyslow5
import torch 
from Simple_Attention import BasicAtt


def plot_squigulator(fasta_folder, blow5_folder, reads_per_sequence):

    
    input_file = fasta_folder
    for j in range (reads_per_sequence):
        output_file = blow5_folder
        # Command to be executed
        if j != 0:
            command = ["/workspaces/MoBa_FP/Squigulator/squigi/squigulator", "-x", "dna-r9-min", input_file, "-o", output_file, "-n", "1"]
        else:
            command = ["/workspaces/MoBa_FP/Squigulator/squigi/squigulator", "-x", "dna-r9-min", input_file, "-o", output_file, "-n", "1", "--ideal"]
        

        # Run the command
        subprocess.run(command, check=True)
        s5 = pyslow5.Open(output_file, 'r')
        reads = s5.seq_reads()
        for read in reads:
            signal = read['signal']
            plt.plot(signal)
        return signal

def create_fasta(sequence, filepath):
    with open(filepath, 'w') as f:
        f.write(f">Test_Fasta\n{sequence}\n")

file = "/workspaces/MoBa_FP/Attention_Test/Files/temp.fasta"
file_blow = "/workspaces/MoBa_FP/Attention_Test/Files/temp.blow5"
sequence = "ACTTAGCAAGCCGACTCAATCACATCTTTCAGCGTGTCATCACATCCTAGCTCATGAAGATGACGTACGTAGCGAAGAGAACGTCTGACGATAGCGCGCGCCTGACGCGGAGATTGGATCGTGTAGTTAGAGTGCTGGACAGCACACGTAGCGGGTACACGTGCAGTGTCACGGAGTCTGTGACAGTGATAATTGTGGCA"
create_fasta(sequence, file)
signal = plot_squigulator(file,file_blow,1)




model = BasicAtt(input_length=1851, tar_length=200,d_model=64)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()



