import torch
import torch.nn as nn
import torch.optim as optim

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
    
    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001, device=None):
        criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_ = []
        accuracy_ = []

        self.train()  # Set model to training mode

        if device:
            self.to(device)
            print("Moved to Device")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

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
                input_lengths = torch.full((outputs.size(1),), self.tar_len *2, dtype=torch.long).to(device)
                #Labels should be (N,S)
                # Compute CTC loss
                loss = criterion(outputs, labels, input_lengths, target_lengths)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                #print(f" Caculated CTC Loss {loss.item()}")

                # Optional: Calculate accuracy
                _, predicted = torch.max(outputs, dim=-1)
                #correct_predictions += (predicted == torch.transpose(labels, 0, 1)).sum().item()
                correct_predictions += 0
                total_predictions += sum(target_lengths).item()  # Total labels for accuracy

            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100 * correct_predictions / total_predictions
            loss_.append(avg_loss)
            accuracy_.append(accuracy)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}% (not implemented)")
            

        print("Training complete!")
        return loss_, accuracy_


