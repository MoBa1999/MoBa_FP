import pyslow5
import h5py
import os
import subprocess


def seq_to_fast5(input_fasta_file, output_fast5,):

    # Definiere den Befehl als Liste
    command = ["seq2squiggle", "predict", input_fasta_file, "-o", "temp.blow5", "-c", "35", "-r", "164", "-v", "debug"]

    try:
        # Ausführen des Befehls und Erfassen der Ausgabe
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # Ausgabe anzeigen
        print("Befehl erfolgreich ausgeführt.")
        print("Ausgabe:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Fehler bei der Ausführung des Befehls:")
        print(e.stderr)
        # Open the .blow5 file using pyslow5
    
    s5 = pyslow5.Open("temp.blow5", 'r')
    reads = s5.seq_reads()
    # Create a new .fast5 file
    with h5py.File(output_fast5, 'w') as fast5_file:
        for read in reads:
            read_id = read['read_id']
            signal = read['signal']
            # Create a group for the read in the .fast5 file
            try:
                read_group = fast5_file.create_group(f'Read_{read_id}')
            except: 
                print("Group coud not be created. Fast5 file loop abborted.")
                print("This is probably due to an already existing file and can be ignored.")
                break
            # Store the signal data
            read_group.create_dataset('Signal', data=signal, dtype='i2')

            # You can add more metadata here if needed
            # Example: read_group.attrs['sampling_rate'] = 4000

            print(f"Processed read_id: {read_id}")

    # Close the .blow5 file
    s5.close()

    print(f"Conversion completed. Output file: {output_fast5}")


input_fasta_file = 'simulator_test.fasta'
output_fast5_file = 'ChandakTest/simu_oligo_1_coverage_35.fast5'
seq_to_fast5(input_fasta_file, output_fast5_file)
