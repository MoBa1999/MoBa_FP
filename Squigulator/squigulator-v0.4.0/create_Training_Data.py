
def process_fasta(input_file, output_prefix):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 2):
        header = lines[i].strip()
        sequence = lines[i+1].strip()

        output_filename = f"{output_prefix}_{i//2}.fasta"
        with open(output_filename, 'w') as out_file:
            out_file.write(f">{header}\n{sequence}\n")

input_file = "oligos_combined.txt"
output_prefix = "Training_Data/training_file"
process_fasta(input_file, output_prefix)