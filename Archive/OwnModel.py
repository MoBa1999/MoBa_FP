import tensorflow as tf
import numpy as np


from Files.ConvBlock import ConvolutionBlock

class Test_Model(tf.keras.Model):
    def __init__(self, d_model, max_pool_layer_idx, num_cnn_blocks, num_classes, label_length):
        super(Test_Model, self).__init__()

        # cnn layer for dimensionality expansion
        self.first_cnn = tf.keras.layers.Conv1D(d_model, 1, padding="same", activation="relu", name="dimensionality-cnn")     
        self.max_pool_layer_idx = max_pool_layer_idx
        self.max_pool = tf.keras.layers.MaxPooling1D(pool_size=2, name="max_pool_1D")
        self.cnn_blocks = [ConvolutionBlock([1, 10, 10, 1], d_model, i) for i in range(num_cnn_blocks)]

        # Update output dimension to match number of classes (including blank)
        self.feed_forward = tf.keras.layers.Dense(num_classes * 200, activation=None, name="feed_forward_layer")  # No activation
        self.reshape_layer = tf.keras.layers.Reshape((200, num_classes), name="reshape_layer")  # Shape should be (timesteps, num_classes)
        self.flatten = tf.keras.layers.Flatten(name="flatten_layer")
        self.softmax_layer = tf.keras.layers.Softmax(axis=-1, name="Normalization")

        #self.label_length = label_length
    def vector_to_base(self, vector):
        """Converts a 4-dimensional vector to a base (A, T, C, G) based on the highest value."""
        max_index = np.argmax(vector)
        if max_index == 0:
            return 'A'
        elif max_index == 1:
            return 'T'
        elif max_index == 2:
            return 'C'
        elif max_index == 3:
            return 'G'
        else:
            raise ValueError("Invalid vector input. Should be a 4-dimensional vector.")

    def get_base_out(self,output_tensor):
        base_sequences = []
        for batch_idx in range(output_tensor.shape[0]):
            base_sequence = ""
            for time_step in range(output_tensor.shape[1]):
                vector = output_tensor[batch_idx, time_step]
                base = self.vector_to_base(vector)
                base_sequence += base
            base_sequences.append(base_sequence)
        return base_sequences
    
    def call(self, inp, print_=False):
        x = self.first_cnn(inp)  
        x = self.call_cnn_blocks(x)
        x = self.flatten(x)
        x = self.feed_forward(x)  
        x = self.reshape_layer(x)
        if print_:  
            print("Shape after reshape_layer:", x.shape)  # Print shape after reshaping
            print(x)
            print("Done")
        x = self.softmax_layer(x)
        return x

    
    def call_cnn_blocks(self, x):
        for i, cnn_block in enumerate(self.cnn_blocks):
            x = cnn_block(x)
            if i in self.max_pool_layer_idx:
                x = self.max_pool(x)
        return x
    
    def call_bases(self,x):
        x = self.call(x)
        x = self.get_base_out(x)
        return x
    
    # def train_step(self, data):
    #     ctc_loss = tf.keras.losses.CTC(name="hi")
    #     inputs, labels = data
   
    #     #tf.print("Inputs shape:", tf.shape(inputs))
    #     #tf.print("Labels shape:", tf.shape(labels))

    #     with tf.GradientTape() as tape:
    #         logits = self(inputs, training=True)  # Forward pass
            
    #         #tf.print(logits)
    #         #tf.print(labels)

            
    #         loss = ctc_loss(labels, logits)
    #     gradients = tape.gradient(loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    #     return {"loss": loss}



    def train(self, training_data, training_labels, epochs=10, batch_size=32):
        # Compile the model with an optimizer
        
        self.compile(optimizer='adam', 
                 loss='categorical_crossentropy', metrics=['accuracy'])
        #self.compile(optimizer=tf.keras.optimizers.Adam())

        # Fit the model to the training data without validation
        #history = self.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)
        
        #print(train_dataset[0].shape)
        history = self.fit(training_data, training_labels, epochs=epochs, batch_size=batch_size)

        return history
