import tensorflow as tf

from Files.ConvBlock import ConvolutionBlock
from Files.Transformer import Transformer


class Test_Model(tf.keras.Model):
    def __init__(self, d_model, max_pool_layer_idx,num_cnn_blocks):
        super(Test_Model, self).__init__()

        # cnn layer for dimensionality expansion
        self.first_cnn = tf.keras.layers.Conv1D(d_model, 1, padding="same", activation="relu", name=f"dimensionality-cnn")
        
        self.max_pool_layer_idx = max_pool_layer_idx
        self.max_pool = tf.keras.layers.MaxPooling1D(pool_size=2, name="max_pool_1D")

        self.cnn_blocks = [ConvolutionBlock([1,3,1], d_model, i) for i in range(num_cnn_blocks)]

    def call(self, inp):
        print("Test")
        x = self.first_cnn(inp) # to bring to proper dimensionality
        x = self.call_cnn_blocks(x) # won't do anything if no cnn blocks
        return x
    
    def call_cnn_blocks(self, x):
        for i,cnn_block in enumerate(self.cnn_blocks):
            x = cnn_block(x)
            
            if(i == self.max_pool_layer_idx):
                x = self.max_pool(x)
        return x
    


    




    