import tensorflow as tf
penis
from Files.ConvBlock import ConvolutionBlock
#from Files.Transformer import Transformer


class Test_Model(tf.keras.Model):
    def __init__(self, d_model, max_pool_layer_idx,num_cnn_blocks):
        super(Test_Model, self).__init__()

        # cnn layer for dimensionality expansion
        self.first_cnn = tf.keras.layers.Conv1D(d_model, 1, padding="same", activation="relu", name=f"dimensionality-cnn")
        
        self.max_pool_layer_idx = max_pool_layer_idx
        self.max_pool = tf.keras.layers.MaxPooling1D(pool_size=2, name="max_pool_1D")

        self.cnn_blocks = [ConvolutionBlock([1,10,10,1], d_model, i) for i in range(num_cnn_blocks)]

        self.feed_forward = tf.keras.layers.Dense(1312, activation='relu', name="feed_forward_layer")
        self.reshape_layer = tf.keras.layers.Reshape((328, 4), name="reshape_layer")
        self.flatten = tf.keras.layers.Flatten(name="flatten_layer")

    def call(self, inp):
        print("Input shape:", inp.shape)  # Print the shape of the input tensor

        # Apply the first convolution layer
        x = self.first_cnn(inp)  
        print("Shape after first_cnn:", x.shape)  # Print the shape after the first Conv1D layer

        # Apply the CNN blocks (if any)
        x = self.call_cnn_blocks(x)  
        print("Shape after call_cnn_blocks:", x.shape)  # Print the shape after CNN blocks

        x = self.flatten(x)
        print("Shape after flatten:", x.shape)  # Print the shape after CNN blocks
        # Apply the feedforward layer
        x = self.feed_forward(x)  
        print("Shape after feed_forward:", x.shape)  # Print the shape after the feedforward layer

        # Reshape to the desired output shape (1, 328, 4)
        x = self.reshape_layer(x)  # Uncomment this line if you want to use a reshape layer
        print("Shape after reshape_layer:", x.shape)  # Print shape after reshaping

        return x
    
    def call_cnn_blocks(self, x):
        for i,cnn_block in enumerate(self.cnn_blocks):
            x = cnn_block(x)
            
            if i in self.max_pool_layer_idx:
                x = self.max_pool(x)
        return x
    def train(self, train_data, train_labels, epochs=10, batch_size=32):
        # Compile the model with an optimizer and loss function
        self.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Fit the model to the training data
        history = self.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

        return history
    


    




    