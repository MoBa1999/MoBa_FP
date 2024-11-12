import torch
import torch.nn as nn
import torch.optim as optim

class MultiSeqModel(nn.Module):
    def __init__(self, input_length, tar_length, conv_1_dim = 10, conv_2_dim = 20,attention_dim =40, n_heads = 10):
        super(MultiSeqModel, self).__init__()

        # 1D Convolutional Layers for each input sequence
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=conv_1_dim, kernel_size=1)
        self.conv1d_2 = nn.Conv1d(in_channels=conv_1_dim, out_channels=10, kernel_size=5, padding = 2)

        # 2D Convolutional Layers for combined input
        self.conv2d_1 = nn.Conv2d(in_channels=10, out_channels=conv_2_dim, kernel_size=(10, 3), padding=(0, 1))
        self.conv2d_2 = nn.Conv2d(in_channels=conv_2_dim, out_channels=attention_dim, kernel_size=(1, 3), padding=(0, 1))
        
        # Max Pooling layer to reduce length by half
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))
        
        # Attention layers
        self.attention1 = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=n_heads)
        self.attention2 = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=n_heads)
        
        # Feedforward layers
        self.fc1 = nn.Linear(int(input_length/2) * attention_dim, int(input_length/2) * 5 )
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
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001, device=None):
        criterion = nn.CrossEntropyLoss()  # Define loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # Define optimizer
        loss_ = []
        accuracy_= []

        self.train()  # Set model to training mode

        # Move the model to the selected device
        if device:
            self.to(device)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for inputs, labels in train_loader:
                # Move data to the selected device
                if device:
                    inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()  # Zero gradients

                outputs = self(inputs)  # Forward pass
                # Reshape outputs and labels to match the dimensions expected by CrossEntropyLoss
                loss = criterion(outputs.view(-1, 4), labels.view(-1, 4).argmax(dim=-1))

                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                epoch_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, dim=-1)  # Predicted classes
                correct_predictions += (predicted == labels.argmax(dim=-1)).sum().item()
                total_predictions += labels.size(0) * labels.size(1)  # Total number of labels in the batch

            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100 * correct_predictions / total_predictions
            loss_.append(avg_loss)
            accuracy_.append(accuracy)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        print("Training complete!")
        return loss_, accuracy_


