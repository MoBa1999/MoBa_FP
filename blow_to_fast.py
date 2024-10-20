import pyslow5
import h5py
import os


def blow5_to_fast5(blow5_file, output_fast5):
    # Open the .blow5 file using pyslow5
    s5 = pyslow5.Open(blow5_file, 'r')

    # Create the .fast5 file
    with h5py.File(output_fast5, 'w') as f5:
        # Iterate over each read in the .blow5 file
        for read_id in s5.read_ids():
            # Get the read data
            read = s5.get_read(read_id)

            # Create a group for each read
            read_group = f5.create_group(f'Reads/{read_id}')

            # Create a dataset for the raw signal
            read_group.create_dataset('Signal', data=read['signal'])

            # Add metadata (e.g., read_id)
            read_group.attrs['read_id'] = read_id.encode()

            # Optionally, add other metadata such as sampling rate, etc.
            # read_group.attrs['sampling_rate'] = read.get('sampling_rate', 'unknown')

            print(f"Added {read_id} to {output_fast5}")

    # Close the .blow5 file
    s5.close()



blow5_file_path = 'path/to/your/file.blow5'
output_fast5_file = 'path/to/output/file.fast5'
blow5_to_fast5(blow5_file_path, output_fast5_file)
