import torch
import torch.nn as nn
import torch.optim as optim
from Levenshtein import distance

class MultiSeqCTCModel(nn.Module):
    def __init__(self, input_length, tar_length, conv_1_dim = 10, conv_2_dim = 20,attention_dim =40,
                  tar_len_multiple=2):
        super(MultiSeqCTCModel, self).__init__()

        # 1D Convolutional Layers for each input sequence
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=conv_1_dim, kernel_size=1)
        self.conv1d_2 = nn.Conv1d(in_channels=conv_1_dim, out_channels=10, kernel_size=5, padding = 2)

        # 2D Convolutional Layers for combined input
        self.conv2d_1 = nn.Conv2d(in_channels=10, out_channels=conv_2_dim, kernel_size=(1, 3), padding=(0, 1))
        self.conv2d_2 = nn.Conv2d(in_channels=conv_2_dim, out_channels=attention_dim, kernel_size=(1, 3), padding=(0, 1))
        
        # Max Pooling layer to reduce length by half
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))
        
        # Attention layers
        self.attention1 = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=10)
        self.attention2 = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=10)
        
        # Feedforward layers
        self.fc1 = nn.Linear(int(input_length/2) * attention_dim, int(input_length/2) * 5 )
        self.fc2 = nn.Linear(int(input_length/2) * 5, 5 * tar_length * tar_len_multiple)

        # Output target length
        self.tar_length = tar_length
        self.tar_len_multiple = tar_len_multiple

        print(self.get_num_params())

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
        
        # Reshape to output shape [batch_size, tar_length *2 (> target_length), 4]
        output = x.view(batch_size, self.tar_length * self.tar_len_multiple, 5)
        
        return output
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001, device=None):
        criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        loss_ = []
        ham_dist_ = []

        self.train()  # Set model to training mode

        if device:
            self.to(device)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            total_samples = 0
            ham_dist = 0

            for inputs, labels in train_loader:
                if device:
                    inputs, labels = inputs.to(device), labels.to(device)

                # Flatten the labels into a single 1D tensor for CTC
                labels = torch.argmax(labels, dim=-1) + 1
                target_lengths = torch.full((labels.size(0),), 200, dtype=torch.long).to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self(inputs).log_softmax(2)

                # Reshape outputs for CTC Loss: (T, N, C)
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full((outputs.size(1),), 400, dtype=torch.long).to(device)

                # Compute CTC loss
                loss = criterion(outputs, labels, input_lengths, target_lengths)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Optional: Calculate accuracy
                test_out = outputs.permute(1, 0, 2)
                for b in range(test_out.shape[0]):
                    col_seq = self.ctc_collapse_probabilities(test_out[b, :, :].cpu().detach().numpy())
                    ham_dist += distance(col_seq, labels[b, :].cpu().detach().numpy())

                # Increment the total number of training samples
                total_samples += inputs.size(0)

            # Step the scheduler
            scheduler.step()

            # Calculate epoch statistics
            avg_loss = epoch_loss / len(train_loader)
            avg_ham_dist = ham_dist / total_samples
            theoretical_accuracy = (self.tar_length - avg_ham_dist) / self.tar_length * 100
            loss_.append(avg_loss)
            ham_dist_.append(avg_ham_dist)

            # Print epoch statistics
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                f"CTC-Loss: {avg_loss:.4f}, "
                f"Ham_Distance: {avg_ham_dist:.2f}, "
                f"Theoretical Accuracy from Hamming: {theoretical_accuracy:.2f}%, "
                f"LR: {scheduler.get_last_lr()[0]:.6f}")

        print("Training complete!")
        return loss_, ham_dist_

