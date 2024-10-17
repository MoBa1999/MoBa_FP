import h5py
import matplotlib.pyplot as plt


def plot_signals_from_hdf5(filename):
    """
    Liest eine HDF5-Datei, plottet jedes Signal und gibt die zugehörige ID aus.

    Args:
        filename (str): Der Pfad zur HDF5-Datei.
    """
    with h5py.File(filename, 'r') as f:
        for test in f:
            raw_signal_data = f[test + '/raw_signal'][:]
            print("Der folgende Read wird analysiert: " +test)

            plt.plot(raw_signal_data,'-')
            sequences_ = extract_sequences()
            #print(sequences_)
            real_seq = find_sequence_by_identifier(sequences_.items(),extract_oligos(test))
            print( real_seq)
            print(len(raw_signal_data)/len(real_seq))
            plt.show()
def get_raw_signal_from_read(filename, read):
    with h5py.File(filename, 'r') as f:
        for test in f:
            if test == read:
                raw_signal_data = f[test + '/raw_signal'][:]
                print("Signal zu folgendem Read wurde gefunden: " +test)
                return raw_signal_data
    print("Kein passendes Signal zum Read gefunden.")
    return [0]

def extract_oligos(read_name, filename="C:/Users/moba/Documents/Uni_München/Forschungspraxis/Overcoming_Nanopore_Data/nanopore_dna_storage_data/decoded_lists/20200814_MIN_0880/exp_0/info.txt"):
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(read_name):
                parts = line.split('\t')
                print("Zugehörige Oligo Sequenz zu Read gefunden: "+ parts[1])
                return parts[1]
    return None

def extract_sequences(fasta_file = "C:/Users/moba/Documents/Uni_München/Forschungspraxis/Overcoming_Nanopore_Data/nanopore_dna_storage_data/oligo_files/oligos_0.fa"):
    sequences = {}
    with open(fasta_file, 'r') as f:
        identifier = None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                identifier = line[1:]  # Remove the '>' character
                sequences[identifier] = ""
            else:
                sequences[identifier] += line
    return sequences
def find_sequence_by_identifier(sequences, identifier):

  for seq_id, sequence in sequences:
    if seq_id == identifier:
        print("Sequence to the identifier was found")
        return sequence
  return None

def find_reads_for_oligo(oligo,filename="C:/Users/moba/Documents/Uni_München/Forschungspraxis/Overcoming_Nanopore_Data/nanopore_dna_storage_data/decoded_lists/20200814_MIN_0880/exp_0/info.txt"):
    reads = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            if parts[1] == oligo:
                print("Zugehöriger Read zu oligo Sequenz gefunden: "+ parts[0])
                reads.append(parts[0])
    if len(reads) == 0:
        print("Keine Reads gefunden!")
    return reads

def plot_all_signals_to_read(signals, oligo_identifier):
    fig, axes = plt.subplots(4, 3, figsize=(18, 20), sharex=True)
    plt.suptitle('Sequenz: ' + oligo_identifier, y=1.0)
    for i, (ax, signal) in enumerate(zip(axes.flat, signals)):
        ax.plot(signal, label=f'Read {i + 1}')
        ax.set_ylabel('Current Value')
        ax.legend(loc='upper right')

    # Hide any unused subplots
    for j in range(len(signals), 12):
        fig.delaxes(axes.flat[j])

    plt.xlabel('Sample Index')

    plt.tight_layout()
    plt.show()


#######Beispielaufruf für das Anzeigen von Oligos
filename = "Example_Data_Nanopore/raw_signal_0.hdf5"
#plot_signals_from_hdf5(filename)


#############Code to search for all available reads for one oligo
oligo_0 = "oligos_0_GCTACATGTATACTGCGAGACAGAC_CGATAGTCGCAGTCGCACATCACTC_3"
sequences_ = extract_sequences()
print(sequences_.items())
#Finds all realted reads
reads = find_reads_for_oligo(oligo_0)
signals = []

for read in reads:
    signals.append(get_raw_signal_from_read(filename,read))
plot_all_signals_to_read(signals, oligo_0)

