import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Levenshtein import distance

class BasicCTC(nn.Module):
    def __init__(self, input_length, tar_length, d_model_at,d_model_conv,classes = 5,max_pool_id = 2, multi_seq_nr = 1, n_heads = 4):
        super(BasicCTC, self).__init__()
        self.d_model_conv = d_model_conv
        self.tar_len = tar_length
        self.classes = classes
        #CNN Layers
        self.first_cnn = nn.Conv1d(in_channels=multi_seq_nr, out_channels=d_model_conv, kernel_size=1, padding=0)
        self.cnn_layer_1 = nn.Conv1d(in_channels=d_model_conv, out_channels=d_model_conv, kernel_size=3, padding="same")
        self.cnn_layer_2 = nn.Conv1d(in_channels=d_model_conv, out_channels=d_model_at, kernel_size=3, padding="same")
        self.first_relu = nn.ReLU()
        self.max_pool_id = max_pool_id
        self.max_pool = nn.MaxPool1d(kernel_size=2)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model_at, nhead=n_heads)
        #Transformer Encoder with multiple layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        #Flatten, Feedforward and Softmax
        self.flatten = nn.Flatten(start_dim=1)  # Flatten starting from channel dimension
        self.fc1 = nn.Linear(d_model_at * int(input_length/2), classes * tar_length *2)
        self.softmax = nn.Softmax(dim=-1)
        print(f"Basic Attention Model with {self.get_num_params()} Paramters created.")

    def forward(self, x):
        #First CNN
        x = self.first_cnn(x)
        x = self.first_relu(x)
        #2 CNN Layer + Maxpool
        x = self.cnn_layer_1(x)
        x = self.max_pool(x)
        x = self.cnn_layer_2(x)

        #Reshape and use self attention
        x = x.permute(2,0,1)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        #FF and output
        x = self.flatten(x)
        x = self.fc1(x)
        #Reshape and Softmax
        x = x.view(-1,self.tar_len *2 ,self.classes)
        x = self.softmax(x)
        return x
    
    
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
    
    
    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001, device=None):
        criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
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
                
                labels = torch.argmax(labels, dim=-1) +1
                target_lengths = torch.tensor([label.size(0) for label in labels], dtype=torch.long).to(device)
                target_lengths = torch.full((labels.size(0),), 200, dtype=torch.long).to(device)
                # Forward pass
                optimizer.zero_grad()
                outputs = self(inputs).log_softmax(2)

                # Reshape outputs for CTC Loss: (T, N, C)
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full((outputs.size(1),), 400, dtype=torch.long).to(device)
                #Labels should be (N,S)
                # Compute CTC loss
                loss = criterion(outputs, labels, input_lengths, target_lengths)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                #print(f" Caculated CTC Loss {loss.item()}")

                # Optional: Calculate accuracy

                test_out = outputs.permute(1,0,2)
                for b in range(test_out.shape[0]):
                    col_seq = self.ctc_collapse_probabilities(test_out[b,:,:].detach().numpy())
                    ham_dist += distance(col_seq,labels[b,:])


                #correct_predictions += (predicted == torch.transpose(labels, 0, 1)).sum().item()
                total_samples += inputs.size(0)

            avg_loss = epoch_loss / len(train_loader)
            avg_ham_dist = ham_dist / total_samples
            theoretical_accuracy = (self.tar_length - avg_ham_dist)/self.tar_length * 100
            loss_.append(avg_loss)
            ham_dist_.append(avg_ham_dist)
            

            print(f"Epoch [{epoch + 1}/{num_epochs}], CTC-Loss: {avg_loss:.4f}, Ham_Distance: {avg_ham_dist:.2f} Theoretical Accuracy from Hamming: {theoretical_accuracy:.2f}%")

        print("Training complete!")
        return loss_, ham_dist_


