from OwnModel import Test_Model
import tensorflow as tf
import numpy as np
from Files.attention_utils import create_combined_mask


print("Creating a Test Model")

# Beispielwert für d_model festlegen
d_model = 5

# Modell instanziieren
model = Test_Model(d_model,1,3)

# Dummy-Eingabe erstellen: z.B. eine Batch-Größe von 3 und 1 Feature mit 23 Zeitschritten
input_tensor = tf.random.normal((3, 24, 1))  # (batch_size, sequence_length, input_channels)
print(input_tensor.numpy())

# Modell aufrufen (Inferenz durchführen)
output = model(input_tensor)

# Ausgabe anzeigen
print("Model output shape:", output.shape)
print("Model output:", output)

