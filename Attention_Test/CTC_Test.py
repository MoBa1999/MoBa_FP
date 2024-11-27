import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Levenshtein import distance as distance
from eval_utils import evaluate_model_ham

class CTC_Test_Model(nn.Module):
    def __init__(self, input_length, tar_length, classes = 5, conv_1_dim = 10, conv_2_dim = 20,attention_dim =40,
                  tar_len_multiple=2, num_reads = 1, n_heads = 8, at_layer =1):
        super(CTC_Test_Model, self).__init__()

        # 1D Convolutional Layers for each input sequence
        self.conv1d_1 = nn.Conv1d(in_channels=num_reads, out_channels=conv_1_dim, kernel_size=1)
        self.conv1d_2 = nn.Conv1d(in_channels=conv_1_dim, out_channels=conv_2_dim, kernel_size=3, padding = 1)
        self.first_relu = nn.ReLU()
        self.conv1d_3 = nn.Conv1d(in_channels=conv_2_dim, out_channels=attention_dim, kernel_size=3, padding = 1)
        # 2D Convolutional Layers for combined input
        
        
        # Max Pooling layer to reduce length by half
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=attention_dim, nhead=n_heads)
        #Transformer Encoder with multiple layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=at_layer)
        
        self.flatten = nn.Flatten(start_dim=1)  # Flatten starting from channel dimension
        # Feedforward layers
        self.fc1 = nn.Linear(int(input_length/2) * attention_dim, classes * tar_len_multiple * tar_length)

        # Output target length
        self.tar_length = tar_length
        self.tar_len_multiple = tar_len_multiple
        self.classes = classes
        print(f"CTC Test Model was initialized with {self.get_num_params()} Parameters.")

    def forward(self, x):
        batch_size, num_sequences, seq_length = x.size()  # Expected shape: [batch_size, 10, seq_length, 1]
        
        # Apply 1D Conv layers on each sequence individually
        x = self.conv1d_1(x)
        x = self.first_relu(x)
        x = self.conv1d_2(x)
        # Max Pool to halve the length
        x = self.max_pool(x)
        x = self.conv1d_3(x)
        
        # Reshape for attention layers
        x = x.permute(2,0,1)  # [reduced_length * num_sequences, batch_size, channels]

        self.transformer(x)
        x = x.permute(1,0,2)
        # Feedforward layers
        x = self.flatten(x)

        x = self.fc1(x)
        
        # Reshape to output shape [batch_size, tar_length *2 (> target_length), 4]
        output = x.view(batch_size, self.tar_length * self.tar_len_multiple, self.classes)
        
        return output
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    def ctc_collapse_probabilities(self,prob_sequence, blank_index=0):
        collapsed_sequence = []
        prev_class = None

        for prob_vector in prob_sequence:
            # Find the index of the class with the highest probability
            current_class = np.argmax(prob_vector)
            
            # Skip if it's a blank or the same as the previous class
            if current_class != blank_index and current_class != prev_class:
                collapsed_sequence.append(current_class)
            prev_class = current_class

        return collapsed_sequence
    
    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001, lr_end=1e-6,
                     device=None, scheduler_type="cosine", test_set = None, save_path = None):
        criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Initialize the chosen scheduler
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_end)
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2, verbose=True)
        elif scheduler_type == "cosine_restart":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=lr_end)
        else:
            raise ValueError(f"Unsupported scheduler_type: {scheduler_type}. Use 'cosine', 'plateau', or 'cosine_restart'.")

        loss_ = []
        ham_dist_ = []
        accs_ = []
        test_accs = [0]

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
                target_lengths = torch.full((labels.size(0),), self.tar_length, dtype=torch.long).to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self(inputs).log_softmax(2)

                # Reshape outputs for CTC Loss: (T, N, C)
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full((outputs.size(1),), self.tar_length*2, dtype=torch.long).to(device)

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
            if scheduler_type == "cosine":
                scheduler.step()
            elif scheduler_type == "plateau":
                scheduler.step(epoch_loss)
            elif scheduler_type == "cosine_restart":
                scheduler.step(epoch)
            if test_set and device:
                test_acc = evaluate_model_ham(self,test_set,device)
                if test_acc > max(test_accs):
                    torch.save(self.state_dict(),save_path)
            # Calculate epoch statistics
            avg_loss = epoch_loss / len(train_loader)
            avg_ham_dist = ham_dist / total_samples
            theoretical_accuracy = (self.tar_length - avg_ham_dist) / self.tar_length * 100
            loss_.append(avg_loss)
            ham_dist_.append(avg_ham_dist)
            accs_.append(theoretical_accuracy)
            test_accs.append(test_acc)

            # Print epoch statistics
            if scheduler_type in ["cosine", "cosine_restart"]:
                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"CTC-Loss: {avg_loss:.4f}, "
                    f"Ham_Distance: {avg_ham_dist:.2f}, "
                    f"Theoretical Accuracy from Levenshtein: {theoretical_accuracy:.2f}%, "
                    f"LR: {scheduler.get_last_lr()[0]:.6f},"
                    f"Test-Lev-Accuracy: {test_acc:.2f}")
            else:  # For 'plateau' or other scheduler types
                print(f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"CTC-Loss: {avg_loss:.4f}, "
                    f"Ham_Distance: {avg_ham_dist:.2f}, "
                    f"Theoretical Accuracy from Levenshtein: {theoretical_accuracy:.2f}%,"
                    f"Test-Lev-Accuracy: {test_acc:.2f}")
            
            if avg_loss <= 0.001:
                print(f"Training completed early! -> Maximum Test Accuracy: {max(test_accs)}")
                return loss_, ham_dist_, accs_, test_accs

        print(f"Training completed early! -> Maximum Test Accuracy: {max(test_accs)}")
        return loss_, ham_dist_, accs_, test_accs


