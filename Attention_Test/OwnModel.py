import tensorflow as tf

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
        self.feed_forward = tf.keras.layers.Dense(num_classes * 328, activation=None, name="feed_forward_layer")  # No activation
        self.reshape_layer = tf.keras.layers.Reshape((328, num_classes), name="reshape_layer")  # Shape should be (timesteps, num_classes)
        self.flatten = tf.keras.layers.Flatten(name="flatten_layer")
        self.label_length = label_length

    def call(self, inp):
        x = self.first_cnn(inp)  
        x = self.call_cnn_blocks(x)
        x = self.flatten(x)
        x = self.feed_forward(x)  
        x = self.reshape_layer(x)  
        print("Shape after reshape_layer:", x.shape)  # Print shape after reshaping
        return x
    
    def call_cnn_blocks(self, x):
        for i, cnn_block in enumerate(self.cnn_blocks):
            x = cnn_block(x)
            if i in self.max_pool_layer_idx:
                x = self.max_pool(x)
        return x

    def ctc_loss(self, y_true, y_pred):
        # y_true: [batch_size, max_time] (actual labels)
        # y_pred: [batch_size, max_time, num_classes] (predictions)
        return tf.reduce_mean(tf.nn.ctc_loss(labels=y_true, logits=y_pred, 
                                              label_length=self.label_length, 
                                              logits_time_major=False))

    def train(self, train_data, train_labels, epochs=10, batch_size=32):
        # Compile the model with an optimizer
        
        self.compile(optimizer='adam', 
                 loss='mse')

        # Fit the model to the training data without validation
        history = self.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

        return history
