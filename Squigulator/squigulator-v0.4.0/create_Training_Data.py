import subprocess
import h5py
import pyslow5
import os
import numpy as np

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

def process_fasta(input_file, output_prefix):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 2):
        header = lines[i].strip()
        sequence = lines[i+1].strip()

        output_filename = f"{output_prefix}_{i//2}.fasta"
        with open(output_filename, 'w') as out_file:
            out_file.write(f">{header}\n{sequence}\n")


def run_squigulator(num_files=100):
    """
    Runs the squigulator command for multiple training files.

    Parameters:
        num_files (int): The number of files to process (default: 100)
    """
    for i in range(num_files + 1):
        input_file = f"Training_Data/training_file_{i}.fasta"
        output_file = f"Training_Data/training_{i}.blow5"
        
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


def blow5_to_pod5(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each .blow5 file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.blow5'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace('.blow5', '.pod5'))

            # Open BLOW5 file
            with pyslow5.Slow5File(input_path, 'r') as blow5_file:
                # Create POD5 file
                with pyslow5.Slow5File(output_path, 'w') as pod5_file:
                    # Copy each read
                    for read in blow5_file.reads():
                        pod5_file.write_read(read)

            print(f"Converted {input_path} to {output_path}")
# Beispielaufruf

def blow5_to_numpy(blow5_dir, output_dir, end_index=100):
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
        fasta_file_path = os.path.join(blow5_dir, f"training_file_{i}.fasta")  # Adjust FASTA file name based on blow5 index
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

                print(sequence_data)
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

input_file = "oligos_combined.txt"
output_prefix = "Training_Data/training_file"

blow5_dir = 'Training_Data/'
output_dir = 'Pod5_Training/'


#Run only for creating fasta files
#process_fasta(input_file, output_prefix)
#run_squigulator()
#blow5_to_fast5_multiple(blow5_dir, output_dir)
#blow5_to_pod5(blow5_dir, output_dir)
blow5_to_numpy(blow5_dir, "Numpy_Data_1/")