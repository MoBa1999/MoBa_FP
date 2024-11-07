import torch
import torch.nn as nn
import torch.optim as optim

class MultiSeqModel(nn.Module):
    def __init__(self, input_length, tar_length):
        super(MultiSeqModel, self).__init__()

        # 1D Convolutional Layers for each input sequence
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=1)
        self.conv1d_2 = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=5, padding = 2)

        # 2D Convolutional Layers for combined input
        self.conv2d_1 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(10, 3), padding=(0, 1))
        self.conv2d_2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(1, 3), padding=(0, 1))
        
        # Max Pooling layer to reduce length by half
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))
        
        # Attention layers
        self.attention1 = nn.MultiheadAttention(embed_dim=40, num_heads=4)
        self.attention2 = nn.MultiheadAttention(embed_dim=40, num_heads=4)
        
        # Feedforward layers
        self.fc1 = nn.Linear(int(input_length/2) * 40, int(input_length/2) * 5 )
        self.fc2 = nn.Linear(int(input_length/2) * 5, 4 * tar_length)

        # Output target length
        self.tar_length = tar_length

    def forward(self, x):
        batch_size, num_sequences, seq_length, channels = x.size()  # Expected shape: [batch_size, 10, seq_length, 1]
        
        # Apply 1D Conv layers on each sequence individually
        sequence_outputs = []
        for seq in range(num_sequences):
            # Select the sequence and reshape to [batch_size, channels, seq_length] for 1D convolution
            x_seq = x[:, seq, :, :].permute(0, 2, 1)  # [batch_size, 1, seq_length]
            x_seq = self.conv1d_1(x_seq)
            x_seq = self.conv1d_2(x_seq)
            sequence_outputs.append(x_seq)
        
        # Stack the sequence outputs along a new dimension for 2D convolution
        x = torch.stack(sequence_outputs, dim=2)  # [batch_size, 20, 10, new_seq_length]
        
        # Apply 2D Convolutions
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)

        # Max Pool to halve the length
        x = self.max_pool(x)
        
        # Reshape for attention layers
        batch_size, channels, num_sequences, reduced_length = x.size()
        x = x.view(batch_size, channels, -1).permute(2, 0, 1)  # [reduced_length * num_sequences, batch_size, channels]

        # Apply attention layers
        x, _ = self.attention1(x, x, x)
        x, _ = self.attention2(x, x, x)

        # Feedforward layers
        x = x.view(batch_size,-1)  # Flatten for linear layers
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Reshape to output shape [batch_size, tar_length, 4]
        output = x.view(batch_size, self.tar_length, 4)
        
        return output
    
    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()  # Define loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # Define optimizer

        self.train()  # Set model to training mode

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Zero gradients
                
                outputs = self(inputs)  # Forward pass
                # Reshape outputs and labels to match the dimensions expected by CrossEntropyLoss
                loss = criterion(outputs.view(-1, 4), labels.view(-1,4).argmax(dim=-1))

                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        print("Training complete!")


