from torch.utils.data import DataLoader, TensorDataset
import torch 
import numpy as np

def get_device(gpu_index=1):
    if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
        device = torch.device(f'cuda:{gpu_index}')  # Use GPU with the specified index
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_index)} with index {gpu_index}")
    else:
        device = torch.device('cpu')  # Fall back to CPU if GPU is not available or invalid index
        print("Using CPU")
    return device

def collapse_string_ctc(input_string):
 
  collapsed_string = ""
  prev_char = None
  for char in input_string:
    if char != "_" and char != prev_char:
      collapsed_string += char
    prev_char = char

  return collapsed_string

def decode_ctc_output(ctc_output):

  decoded_sequence = ""
  for time_step_output in ctc_output:
    max_index = np.argmax(time_step_output)
    if max_index == 0:
      decoded_sequence += "_"
    elif max_index == 1:
      decoded_sequence += "A"
    elif max_index == 2:
      decoded_sequence += "T"
    elif max_index == 3:
      decoded_sequence += "C"
    elif max_index == 4:
      decoded_sequence += "G"

  return decoded_sequence

def vector_to_base(vector):
    vector = vector.tolist()
    """Konvertiert einen 4-dimensionalen Vektor in die entsprechende Base (A, T, C, G)."""
    if vector == [1, 0, 0, 0]:
        return 'A'
    elif vector == [0, 1, 0, 0]:
        return 'T'
    elif vector == [0, 0, 1, 0]:
        return 'C'
    elif vector == [0, 0, 0, 1]:
        return 'G'
    else:
        return None  # Falls der Vektor ungültig ist

def vectors_to_sequence(vectors):
    """Konvertiert eine Liste von 4-dimensionalen Vektoren in eine DNA-Sequenz als String."""
    sequence = ''.join([vector_to_base(vec) for vec in vectors])
    return sequence

def get_data_loader(data_path_numpy, end_sequence, batch_size=16, start_sequence=0, overwrite_max_length=None, dim_squeeze=False, num_reads=1):
    signals = []
    seqs = []

    # Find the maximum length across all signals for padding
    max_length = 0
    for i in range(start_sequence, end_sequence):
        try:
            # Load all signals for the sequence
            sequence_signals = np.load(f"{data_path_numpy}/signal_seq_{i}.npy")
            # Find the longest signal in the current sequence
            max_length = max(max_length, max(signal.shape[0] for signal in sequence_signals))
        except FileNotFoundError:
            print(f"{data_path_numpy}/signal_seq_{i}.npy - Signal file not found, skipping sequence.")
    
    print(f"{max_length} is the longest length of a read in the dataset with {end_sequence - start_sequence} sequences.")

    if overwrite_max_length:
        if overwrite_max_length > max_length:
            print("Max Length is overwritten.")
            max_length = overwrite_max_length

    # Load, pad, and store signals and sequences
    for i in range(start_sequence, end_sequence):
        try:
            # Load target sequence
            seq = np.load(f"{data_path_numpy}/signal_seq_{i}_tarseq.npy")
        except FileNotFoundError:
            print(f"{data_path_numpy}/signal_seq_{i}_tarseq.npy - Target sequence file not found, skipping sequence.")
            continue

        try:
            # Load all signals for the sequence
            sequence_signals = np.load(f"{data_path_numpy}/signal_seq_{i}.npy")
            sequence_signals = sequence_signals[0:num_reads,:]
        except FileNotFoundError:
            print(f"{data_path_numpy}/signal_seq_{i}.npy - Signal file not found, skipping sequence.")
            continue

        # Normalize and pad each signal in the sequence
        padded_signals = []
        for signal in sequence_signals:
            mean = np.mean(signal)
            std = np.std(signal)
            normalized_signal = (signal - mean) / std

            # Pad to max_length
            padding_length = max_length - normalized_signal.shape[0]
            padded_signal = np.pad(normalized_signal, (0, padding_length), mode='constant', constant_values=0)
            padded_signals.append(padded_signal)

        # Add the padded signals and target sequence to the respective lists
        signals.append(padded_signals)
        seqs.append(seq)

    # Convert lists to tensors
    signals = torch.from_numpy(np.array(signals)).float()  # Shape: (num_sequences, num_reads, max_length)
    seqs = torch.from_numpy(np.array(seqs)).float()  # Shape: (num_sequences, sequence_length, feature_dim)

    # Adjust the signal tensor shape to match expected dimensions
    signals = signals.unsqueeze(-1)  # Add a channel dimension: (num_sequences, num_reads, max_length, 1)
    if dim_squeeze:
        signals = signals.squeeze(3)  # Remove the last dimension if requested

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(signals, seqs)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return max_length, train_loader

def get_data_loader_real(data_path_numpy,end_sequence,batch_size = 16,start_sequence = 0, num_reads = 10, overwrite_max_length = None, dim_squeeze=False):
    signals = []
    seqs = []


    # Find the maximum length across all signals for padding
    max_length = 0
    max_seq_len = 0
    for i in range(start_sequence,end_sequence):
        try:
          seq = np.load(f"{data_path_numpy}/signal_seq_{i}_read_{0}_tarseq.npy")
        except:
           print(f"{data_path_numpy}/signal_seq_{i}_read_{0}_tarseq.npy - Sequence not found")
           continue
        max_seq_len = max(max_seq_len, seq.shape[0])
        for j in range(num_reads):
            try:
              signal = np.load(f"{data_path_numpy}/signal_seq_{i}_read_{j}.npy")
              max_length = max(max_length, signal.shape[0])
            except:
               print(f"{data_path_numpy}/signal_seq_{i}_read_{j}.npy - Could not be found, going on with next")
    print(f"{max_length} is the longest length of a read for dataset with {end_sequence-start_sequence} Sequences, longest Sequenz: {max_seq_len}")

    if overwrite_max_length:
        if overwrite_max_length > max_length:
            print("Max Length is overwritten")
            max_length = overwrite_max_length

    # Load, pad, and store signals and sequences
    for i in range(start_sequence, end_sequence):
        #List initialized
        sequence_signals = []

        try:
          seq = np.load(f"{data_path_numpy}/signal_seq_{i}_read_{0}_tarseq.npy")
        except:
           print(f"{data_path_numpy}/signal_seq_{i}_read_{0}_tarseq.npy - Sequence not found")
           continue
        for j in range(num_reads):
            try:
              # Load signal and pad to max_length
              signal = np.load(f"{data_path_numpy}/signal_seq_{i}_read_{j}.npy")
            except:
               print(f"{data_path_numpy}/signal_seq_{i}_read_{j}.npy - Data not found.")
               continue
            # Normalize the signal
            mean = np.mean(signal)
            std = np.std(signal)
            normalized_signal = (signal - mean) / std
            
            # Pad to max_length
            padding_length = max_length - normalized_signal.shape[0]
            padded_signal = np.pad(normalized_signal, (0, padding_length), mode='constant', constant_values=0)

            pad_len_seq = max_seq_len - seq.shape[0]
            seq_padded = np.pad(seq, ((0, pad_len_seq), (0, 0)), mode='constant', constant_values=0)
            sequence_signals.append(padded_signal)
            
        # Load target sequence
        signals.append(sequence_signals)       
        seqs.append(seq_padded)

    # Convert lists to arrays
    signals = torch.from_numpy(np.array(signals))
    seqs = torch.from_numpy(np.array(seqs))
    signals = signals.view(signals.shape[0], signals.shape[1], signals.shape[2], 1).float()
    if dim_squeeze:
        signals = signals.squeeze(3)
    dataset = TensorDataset(signals, seqs)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return max_length, train_loader
