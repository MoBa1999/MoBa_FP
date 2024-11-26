import torch
import torch.nn as nn
import torch.optim as optim
from Levenshtein import distance
from eval_utils import evaluate_model
class BasicAtt(nn.Module):
    def __init__(self, input_length, tar_length, d_model,classes = 4,max_pool_id = 2, multi_seq_nr = 1, n_heads = 4):
        super(BasicAtt, self).__init__()
        self.d_model = d_model
        self.tar_len = tar_length

        #CNN Layers
        self.first_cnn = nn.Conv1d(in_channels=multi_seq_nr, out_channels=d_model, kernel_size=1, padding=0)
        self.cnn_layer_1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding="same")
        self.cnn_layer_2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding="same")
        self.first_relu = nn.ReLU()
        self.max_pool_id = max_pool_id
        self.max_pool = nn.MaxPool1d(kernel_size=2)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        #Transformer Encoder with multiple layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        #Flatten, Feedforward and Softmax
        self.flatten = nn.Flatten(start_dim=1)  # Flatten starting from channel dimension
        self.fc1 = nn.Linear(d_model * int(input_length/2), classes * tar_length)
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
        x = x.view(-1,self.tar_len,4)
        x = self.softmax(x)
        return x
    
    
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001,
                     device=None, test_set= None, save_path = None, lr_end = 0.00001):
        criterion = nn.CrossEntropyLoss()  # Define loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # Define optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_end)
        loss_ = []
        accuracy_= []
        ham_acc_ = []
        test_acc_ = [0]

        self.train()  # Set model to training mode

        # Move the model to the selected device
        if device:
            print("Moved to Device")
            self.to(device)
            # for block in self.cnn_blocks:
            #     block.to(device)

        for epoch in range(num_epochs):
            ham_loss = 0
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

                for b in range(predicted.shape[0]):
                    ham_loss+= distance(predicted[b,:].cpu().detach().numpy(),labels.argmax(dim=-1)[b,:].cpu().detach().numpy())

            if test_set:
                _,_,test_ac = evaluate_model(self,test_set,criterion,device)
                if test_ac > max(test_acc_):
                    torch.save(self.state_dict(),save_path)
                test_acc_.append(test_ac)

            scheduler.step()    

            avg_ham = ham_loss/ (total_predictions/ self.tar_len)
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100 * correct_predictions / total_predictions
            ham_ac = (self.tar_len - avg_ham)/self.tar_len * 100
            loss_.append(avg_loss)
            accuracy_.append(accuracy)
            ham_acc_.append(ham_ac)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%,  Lev-Accuracy: {ham_ac:.2f}% ,LR: {scheduler.get_last_lr()[0]:.6f},  Lev-Test-Accuracy: {test_ac:.2f}%")

        print("Training complete!")
        return loss_, accuracy_,ham_acc_, test_acc_


