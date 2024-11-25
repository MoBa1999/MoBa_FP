from torch.utils.data import DataLoader, TensorDataset
import torch 
import numpy as np

def get_device(gpu_index=1):
    if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
        device = torch.device(f'cuda:{gpu_index}')  # Use GPU with the specified index
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_index)}")
    else:
        device = torch.device('cpu')  # Fall back to CPU if GPU is not available or invalid index
        print("Using CPU")
    return device
def vector_to_base(vector):
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

def get_data_loader(data_path_numpy,end_sequence,batch_size = 16,start_sequence = 0, num_reads = 10):
    signals = []
    seqs = []


    # Find the maximum length across all signals for padding
    max_length = 0
    for i in range(start_sequence,end_sequence):
        for j in range(num_reads):
            signal = np.load(f"{data_path_numpy}/signal_seq_{i}_read_{j}.npy")
            max_length = max(max_length, signal.shape[0])
    print(f"{max_length} is the longest length of a read")

    # Load, pad, and store signals and sequences
    for i in range(start_sequence, end_sequence):
        #List initialized
        sequence_signals = []
        for j in range(num_reads):
            # Load signal and pad to max_length
            signal = np.load(f"{data_path_numpy}/signal_seq_{i}_read_{j}.npy")
            # Normalize the signal
            mean = np.mean(signal)
            std = np.std(signal)
            normalized_signal = (signal - mean) / std
            
            # Pad to max_length
            padding_length = max_length - normalized_signal.shape[0]
            padded_signal = np.pad(normalized_signal, (0, padding_length), mode='constant', constant_values=0)
            sequence_signals.append(padded_signal)
            
        # Load target sequence
        signals.append(sequence_signals)
        seq = np.load(f"{data_path_numpy}/signal_seq_{i}_read_{0}_tarseq.npy")
        seqs.append(seq)

    # Convert lists to arrays
    signals = torch.from_numpy(np.array(signals))
    seqs = torch.from_numpy(np.array(seqs))
    signals = signals.view(signals.shape[0], signals.shape[1], signals.shape[2], 1).float()
    if num_reads == 1:
        signals = signals.squeeze(3)
    dataset = TensorDataset(signals, seqs)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return max_length, train_loader