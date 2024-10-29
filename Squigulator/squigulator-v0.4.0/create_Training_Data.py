import subprocess
import h5py
import pyslow5
import os
from blue_crab import Slow5ToPod5


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



def blow5_to_pod5_multiple(blow5_dir, output_dir):
  """
  Converts multiple blow5 files in a directory to pod5 format.

  Args:
      blow5_dir (str): Path to the directory containing blow5 files.
      output_dir (str): Path to the directory where pod5 files will be saved.
  """
  
  # List all files in the blow5 directory
  blow5_files = [f for f in os.listdir(blow5_dir) if f.endswith('.blow5')]

  # Process all .blow5 files
  for blow_file in blow5_files:
    blow_file_path = os.path.join(blow5_dir, blow_file)
    output_pod5 = os.path.join(output_dir, os.path.splitext(blow_file)[0] + '.pod5')  # Remove extension and add .pod5

    # Convert using Slow5ToPod5 converter
    converter = Slow5ToPod5(blow_file_path, output_pod5)
    converter.convert()

    print(f"Conversion completed for file {blow_file}. Output file: {output_pod5}")

# Example usage
blow5_dir = "/path/to/your/blow5/files"
output_dir = "/path/to/output/pod5/files"
blow5_to_pod5_multiple(blow5_dir, output_dir)

# Beispielaufruf

input_file = "oligos_combined.txt"
output_prefix = "Training_Data/training_file"

blow5_dir = 'Training_Data/'
output_dir = 'Fast5_Training'


#Run only for creating fasta files
#process_fasta(input_file, output_prefix)
#run_squigulator()
blow5_to_fast5_multiple(blow5_dir, output_dir)