from OwnModel import Test_Model
import tensorflow as tf
import numpy as np
from Files.attention_utils import create_combined_mask
import matplotlib.pyplot as plt

def collapse_sequences(sequences):
    collapsed = []
    for seq in sequences:
        # Filter out zeros and collapse consecutive duplicates
        filtered_seq = [x for x in seq if x != 0]
        collapsed_seq = [filtered_seq[0]] if filtered_seq else []
        
        for i in range(1, len(filtered_seq)):
            if filtered_seq[i] != filtered_seq[i - 1]:
                collapsed_seq.append(filtered_seq[i])
        
        collapsed.append(collapsed_seq)

    # Determine the max length for padding
    max_length = max(len(seq) for seq in collapsed)

    # Pad sequences to the right with zeros
    padded = np.array([seq + [0] * (max_length - len(seq)) for seq in collapsed])
    
    return padded

# Custom CTC loss function
Test_Object = tf.keras.losses.CTC(name="Customized")
print(Test_Object.get_config())

# Define parameters
batch_size = 2
max_length = 10
num_classes = 2

# Create y_true (integer labels) with values in range [0, num_classes - 1]
#y_true = np.random.randint(1, num_classes, size=(batch_size, max_length), dtype=int)
y_true = np.array([[1, 2],[1 ,2]])

print("y_true:", y_true)

# Create y_pred (logits) with high values in places that match y_true
#y_pred = np.full((batch_size, max_length, num_classes), -100.0)
# y_pred  = np.array([[
#     [-100, 100, -100],
#     [-100, 100, -100],
#     [-100, -100, 100],
#     [-100, -100, 100],
# ], [
#     [-100, 100, -100],
#     [100, -100, -100],
#     [-100, -100, 100],
#     [-100, -100, 100],
# ]])
y_pred = np.random.uniform(low=-100, high=100, size=(2, 5, 3))
print("y_pred:", y_pred)

y_true_c = collapse_sequences(np.argmax(y_pred, axis=2))
print(y_true_c)

print(f"calculated y_true: {y_true_c}")
print(f"Real y_true: {y_true}")
# Calculate CTC Loss
loss = Test_Object(y_true_c, y_pred)
print(f"DER FUCKING CTC LOSS: {loss}")


print("Check mit Mean Squared Error")
# Erstelle Beispiel-Inputs, wobei Vorhersagen und tatsächliche Werte identisch sind
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1, 2, 3, 4, 5])

# Definiere den MSE
mse = tf.keras.losses.MeanSquaredError()

# Berechne den MSE
result = mse(y_true, y_pred)
print(result.numpy())  # Ausgabe: tf.Tensor(0.0, shape=(), dtype=float32)
