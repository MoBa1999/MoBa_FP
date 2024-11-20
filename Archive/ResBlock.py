import torch
import torch.nn as nn

class ResiBlock(nn.Module):
    def __init__(self, cnn_dims, filters, idx):
        super(ResiBlock, self).__init__()
        
        self.cnn_layers = nn.ModuleList([nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=dim, padding="same") for dim in cnn_dims])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(filters) for _ in cnn_dims])
        self.activation_layer = nn.ReLU()

    def forward(self, x):
        res = x
        for cnn_layer, bn_layer in zip(self.cnn_layers, self.bn_layers):
            x = cnn_layer(x)
            x = bn_layer(x)
        
        x += res
        x = self.activation_layer(x)
        return x