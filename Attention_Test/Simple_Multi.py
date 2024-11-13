import torch
import torch.nn as nn
import torch.optim as optim

class BasicMulti(nn.Module):
    def __init__(self, input_length, tar_length, d_model,classes = 4,max_pool_id = 2, multi_seq_nr = 2):
        super(BasicMulti, self).__init__()
        self.d_model = d_model
        self.tar_len = tar_length
        self.first_cnn = nn.Conv1d(in_channels=multi_seq_nr, out_channels=d_model, kernel_size=1, padding=0)
        self.cnn_layer = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding="same")
        self.first_relu = nn.ReLU()
        self.max_pool_id = max_pool_id
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        #self.cnn_blocks = [ResiBlock([1,3,1], d_model, i) for i in range(cnn_blocks)]
        self.flatten = nn.Flatten(start_dim=1)  # Flatten starting from channel dimension
        self.fc1 = nn.Linear(d_model * int(input_length/2), classes * tar_length)
        self.softmax = nn.Softmax(dim=-1)
        print(f"Basic Model with {self.get_num_params()} Paramters created.")

    def forward(self, x):
        #First CNN
        x = self.first_cnn(x)
        x = self.first_relu(x)
        #2 CNN Layer + Maxpool
        x = self.cnn_layer(x)
        x = self.max_pool(x)
        x = self.cnn_layer(x)
        #FF and output
        x = self.flatten(x)
        x = self.fc1(x)
        #Reshape and Softmax
        x = x.view(-1,self.tar_len,4)
        x = self.softmax(x)
        return x
    
    
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
            print("Moved to Device")
            self.to(device)
            # for block in self.cnn_blocks:
            #     block.to(device)

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
                loss = criterion(outputs.float(), labels.float())
                #print(outputs.float())
                #print(labels.float())
                #print(f"Current loss in batch: {loss}")

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


