import pyslow5
import h5py
import os
import subprocess


def seq_to_fast5(blow_file, output_fast5,):

    
    s5 = pyslow5.Open(blow_file, 'r')
    reads = s5.seq_reads()
    # Create a new .fast5 file
    with h5py.File(output_fast5, 'w') as fast5_file:
        for read in reads:
            read_id = read['read_id']
            signal = read['signal']
            print("Length of Read" + str(read_id) + " ist " + str(len(signal)))
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


blow_file = 'test.blow5'
output_fast5_file = 'Squil_Test/test_1.fast5'
seq_to_fast5(blow_file, output_fast5_file)
