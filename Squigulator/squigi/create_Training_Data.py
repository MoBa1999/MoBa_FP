import subprocess
import h5py
import pyslow5
import os
import numpy as np
import random 
import string

def base_to_vector(base):
        """Konvertiert eine Base (A, T, C, G) in einen 4-dimensionalen Vektor."""
        vector = [0, 0, 0, 0]
        if base == 'A':
            vector[0] = 1
        elif base == 'T':
            vector[1] = 1
        elif base == 'C':
            vector[2] = 1
        elif base == 'G':
            vector[3] = 1
        return vector

def process_fasta(input_file, output_prefix, max_files=100):
    """
    Processes a FASTA file and splits it into multiple FASTA files based on a given maximum number.

    Args:
        input_file (str): Path to the input FASTA file.
        output_prefix (str): Prefix for the output file names.
        max_files (int): Maximum number of output files to generate.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 2):
        if i // 2 >= max_files:  # Check if we've reached the maximum number of files
            break
        header = lines[i].strip()
        sequence = lines[i+1].strip()
        output_filename = f"{output_prefix}_{i//2}.fasta"
        with open(output_filename, 'w') as out_file:
            out_file.write(f">{header}\n{sequence}\n")


def run_squigulator(folder, num_files=100):
    """
    Runs the squigulator command for multiple training files.

    Parameters:
        num_files (int): The number of files to process (default: 100)
    """
    for i in range(num_files - 1):
        input_file = f"{folder}/Rd_Data_Fasta/fasta_file_{i}.fasta"
        output_file = f"{folder}/Rd_Data_Blow5/training_{i}.blow5"
        
        # Command to be executed
        command = ["./squigulator", "-x", "dna-r9-min", input_file, "-o", output_file, "-n", "1"]
        
        # Run the command
        subprocess.run(command, check=True)

def blow5_to_fast5_multiple(blow5_dir, output_dir):
       
    # Liste aller Dateien im blow5-Verzeichnis abrufen
    blow5_files = [f for f in os.listdir(blow5_dir) if f.endswith('.blow5')]
    
    # Alle .blow5-Dateien verarbeiten
    for i, blow_file in enumerate(blow5_files, start=1):
        blow_file_path = os.path.join(blow5_dir, blow_file)
        output_fast5 = os.path.join(output_dir, f'training_fast_{i}.fast5')
        
        s5 = pyslow5.Open(blow_file_path, 'r')
        reads = s5.seq_reads()
        
        # Erstellen einer neuen .fast5 Datei
        with h5py.File(output_fast5, 'w') as fast5_file:
            for read in reads:
                read_id = read['read_id']
                signal = read['signal']
                print("Length of Read " + str(read_id) + " ist " + str(len(signal)))
                
                # Erstellen einer Gruppe für den Read in der .fast5 Datei
                try:
                    read_group = fast5_file.create_group(f'Read_{read_id}')
                except:
                    print("Group could not be created. Fast5 file loop aborted.")
                    print("This is probably due to an already existing file and can be ignored.")
                    break
                
                # Speichern der Signal-Daten
                read_group.create_dataset('Signal', data=signal, dtype='i2')
                
                # Beispiel für zusätzliche Metadaten (falls benötigt)
                # read_group.attrs['sampling_rate'] = 4000
                
                print(f"Processed read_id: {read_id}")
        
        # Schließen der .blow5 Datei
        s5.close()
        print(f"Conversion completed for file {blow_file}. Output file: {output_fast5}")


def blow5_to_pod5(blow5_dir, output_dir, end_index=100):
    """
    Converts multiple blow5 files in a directory to separate pod5 files.

    Args:
        blow5_dir (str): Path to the directory containing blow5 files.
        output_dir (str): Path to the directory where pod5 files will be saved.
        end_index (int): End index for file numbering (exclusive).
    """
    # List all files in the blow5 directory
    blow5_files = [f"training_{i}.blow5" for i in range(end_index)]

    # Process all .blow5 files
    for i, blow_file in enumerate(blow5_files, start=0):
        blow_file_path = os.path.join(blow5_dir, blow_file)
        
        # Create output filename for the pod5 file
        pod5_file_path = os.path.join(output_dir, f'training_{i}.pod5')

        # Open the blow5 file
        s5 = pyslow5.Open(blow_file_path, 'r')
        reads = s5.seq_reads()

        # Create a new pod5 file to save the converted reads
        pod5 = pyslow5.Open(pod5_file_path, 'w')

        # Process each read
        for read in reads:
            read_id = read['read_id']
            signal = read['signal']

            # Convert signal to NumPy array
            signal_array = np.array(signal, dtype=np.float32)  # Change dtype as needed

            # Add the read to the pod5 file
            pod5.add_read(read_id, signal_array)

            print(f"Converted read {read_id} from {blow_file_path} to {pod5_file_path}")

        # Close the blow5 and pod5 files
        s5.close()
        pod5.close()

    print("Conversion completed!")
# Beispielaufruf

def blow5_to_numpy(blow5_dir, output_dir, fasta_dir, end_index=100):
    """
    Converts multiple blow5 files in a directory to separate NumPy arrays and saves them as .npy files,
    additionally analyzing the corresponding FASTA files and saving the second line as a separate NumPy array.

    Args:
        blow5_dir (str): Path to the directory containing blow5 files.
        output_dir (str): Path to the directory where .npy files will be saved.
        end_index (int): End index for file numbering (exclusive).
    """
    # List all files in the blow5 directory
    blow5_files = [f"training_{i}.blow5" for i in range(end_index)]

    # Process all .blow5 files
    for i, blow_file in enumerate(blow5_files, start=0):
        blow_file_path = os.path.join(blow5_dir, blow_file)
        fasta_file_path = os.path.join(fasta_dir, f"fasta_file_{i}.fasta")  # Adjust FASTA file name based on blow5 index
    #TODO: CHECK IF ASSIGNMENT IS CORRECT

        # Open the blow5 file
        s5 = pyslow5.Open(blow_file_path, 'r')
        reads = s5.seq_reads()

        # Process each read
        for read in reads:
            read_id = read['read_id']
            signal = read['signal']

            # Convert signal to NumPy array
            signal_array = np.array(signal, dtype=np.int16)

            # Read and process FASTA file (assuming second line contains sequence)
            try:
                with open(fasta_file_path, 'r') as fasta_file:
                    lines = fasta_file.readlines()
                    sequence = lines[1].strip()  # Second line (assuming sequence)

              
                sequence_data = []  # Create an empty list to store the vectors
                for x in sequence:
                    sequence_data.append(base_to_vector(x))
                # Convert the list of vectors to a NumPy array
                sequence_data = np.array(sequence_data)

                #print(sequence_data)
                # Combine signal and sequence arrays
                

                # Create output filename for the combined array
                output_npy = os.path.join(output_dir, f'signal_{i}.npy')
                output_tar = os.path.join(output_dir, f'signal_{i}_tarseq.npy')
                
                # Save the combined array as a NumPy file
                np.save(output_npy, signal_array)
                np.save(output_tar, sequence_data)
                print(signal_array)
                print(f"Saved read {read_id} to {output_npy}")
            except FileNotFoundError:
                print(f"FASTA file not found: {fasta_file_path}")

        s5.close()



def generate_fasta_files(num_files, output_folder, sequences_per_file, bias=0.5):
    """
    Generates multiple FASTA files with random sequences, with an optional bias against repeated bases.

    Args:
        num_files: The number of FASTA files to create.
        output_folder: The directory to save the FASTA files.
        sequences_per_file: The number of sequences per file.
        bias: The bias factor for repeated bases. A value of 0.5 means half the likelihood of repeated bases.
    """

    def generate_random_sequence(length, bias):
        def biased_choice(last_base):
            weights = [0.25, 0.25, 0.25, 0.25]
            if last_base is not None:
                weights['ATGC'.index(last_base)] *= bias
                weights = [w / sum(weights) for w in weights]  # Normalize weights
            return random.choices('ATGC', weights=weights)[0]

        sequence = ""
        for _ in range(length):
            sequence += biased_choice(sequence[-1] if sequence else None)
        return sequence

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(num_files):
        filename = f"fasta_file_{i}.fasta"
        filepath = os.path.join(output_folder, filename)

        with open(filepath, 'w') as f:
            for j in range(sequences_per_file):
                sequence = generate_random_sequence(200, bias)
                f.write(f">File_{i}_Seq_{j}\n{sequence}\n")


num_files = 10000
fasta_dir = "/media/hdd1/MoritzBa/Rd_Data_Fasta"
output_folder = "/media/hdd1/MoritzBa/Rd_Data_Fasta"
sequence_per_file = 1
#generate_fasta_files(num_files,output_folder, sequence_per_file)

#input_oligos = "oligos_combined.txt"
#output_prefix = "Tr_Data_Fasta/batch_0"

blow5_dir = '/media/hdd1/MoritzBa/Rd_Data_Blow5'
output_dir = '/media/hdd1/MoritzBa/Rd_Data_Numpy'


#Run only for creating fasta files
#process_fasta(input_oligos, output_prefix)
#run_squigulator("/media/hdd1/MoritzBa",num_files=10000)
blow5_to_numpy(blow5_dir, output_dir, fasta_dir, end_index=9998)
#blow5_to_pod5(blow5_dir, output_dir)
#blow5_to_pod5(blow5_dir, "Pod5_Training/")

