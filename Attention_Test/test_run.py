from OwnModel import Test_Model
import tensorflow as tf
import numpy as np
from Files.attention_utils import create_combined_mask
import matplotlib.pyplot as plt

#################################Konsolen Outputs aus von Tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


signals = []
seqs = []
desired_length = 2795
#Load Data 
for i in range(100):
    signal = np.load(f"/workspaces/MoBa_FP/Squigulator/squigi/Tr_Data_Numpy/signal_{i}.npy")

    #Padding 
    if signal.shape[0] < desired_length:
        # Berechnung der Anzahl der Nullen, die hinzugefügt werden müssen
        padding_length = desired_length - signal.shape[0]
        # Signal mit Nullen auffüllen (pad_width ist ein Tupel von (vorne, hinten))
        signal = np.pad(signal, (0, padding_length), mode='constant', constant_values=0)
    signals.append(signal)
    seq = np.load(f"/workspaces/MoBa_FP/Squigulator/squigi/Tr_Data_Numpy/signal_{i}_tarseq.npy")  
    seqs.append(seq)
seqs = np.array(seqs)
signals = np.array(signals)


# Beispielwert für d_model festlegen
d_model = 256
model = Test_Model(d_model,[1,2],4,5, 328)

#Testing:
input_tensor_example = tf.convert_to_tensor(signal.reshape((1, len(signals[0]), 1)), dtype=tf.float32)

#Converting Old
#signals = tf.convert_to_tensor(signals.reshape(signals.shape[0], signals.shape[1], 1), dtype=tf.float32)
#seqs = tf.convert_to_tensor(seqs.reshape(seqs.shape[0], seqs.shape[1], 4), dtype=tf.float32)


#Converting CTC
# Konvertiere in Tensoren und forme für TensorFlow um
signals = tf.convert_to_tensor(signals.reshape(signals.shape[0], signals.shape[1], 1), dtype=tf.float32)  # Form: (100, 2795, 1)
seqs = tf.convert_to_tensor(seqs, dtype=tf.int32)  # Sequenzen als integer-Indizes für CTC Loss
batch_size = 100
train_dataset = tf.data.Dataset.from_tensor_slices((signals, seqs))
train_dataset = train_dataset.shuffle(buffer_size=100).batch(batch_size)
train_dataset = train_dataset.map(lambda x, y: (x, tf.argmax(y, axis=-1, output_type=tf.int32)+1))

# Modell aufrufen (Inferenz durchführen)
for batch in train_dataset.take(1):
    inputs, labels = batch
    logits = model(inputs)
    #print(logits)
    #print(labels)
    

#Trainings Test
history = model.train(train_dataset, epochs=50, batch_size=2)
#print("Training completed!")
print(history.history['loss'])
#print(history.history['accuracy'])
print("Example")
#print(model.call_bases(input_tensor_example))


