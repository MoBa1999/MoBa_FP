from OwnModel import Test_Model
import tensorflow as tf
import numpy as np
from Files.attention_utils import create_combined_mask
import matplotlib.pyplot as plt

# Beispielwert für d_model festlegen
d_model = 256
model = Test_Model(d_model,[1,2],4,4, 328)

signals = []
seqs = []
desired_length = 2795
#Load Data 
for i in range(100):
    signal = np.load(f"/workspaces/MoBa_FP/Squigulator/squigulator-v0.4.0/Numpy_Data_1/signal_{i}.npy")

    #Padding 
    if signal.shape[0] < desired_length:
        # Berechnung der Anzahl der Nullen, die hinzugefügt werden müssen
        padding_length = desired_length - signal.shape[0]
        # Signal mit Nullen auffüllen (pad_width ist ein Tupel von (vorne, hinten))
        signal = np.pad(signal, (0, padding_length), mode='constant', constant_values=0)
    signals.append(signal)
    seq = np.load(f"/workspaces/MoBa_FP/Squigulator/squigulator-v0.4.0/Numpy_Data_1/signal_{i}_tarseq.npy")  
    seqs.append(seq)
seqs = np.array(seqs)
signals = np.array(signals)

#Testing:
input_tensor_example = tf.convert_to_tensor(signal.reshape((1, len(signal), 1)), dtype=tf.float32)

signals = tf.convert_to_tensor(signals.reshape(signals.shape[0], signals.shape[1], 1), dtype=tf.float32)
seqs = tf.convert_to_tensor(seqs.reshape(seqs.shape[0], seqs.shape[1], 4), dtype=tf.float32)
# Modell aufrufen (Inferenz durchführen)


#Trainings Test
history = model.train(signals, seqs, epochs=20, batch_size=32)
print("Training completed!")
print(history.history['loss'])
print("Example")
print(model(input_tensor_example))


