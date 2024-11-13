import torch 
import matplotlib.pyplot as plt
import os 
import numpy as np

def plot_training_curves(losses, accuracies):
    """Plots training loss and accuracy curves.

    Args:
        losses: A list of training losses.
        accuracies: A list of training accuracies.
    """

    plt.figure(figsize=(10, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training  Loss')
    plt.legend()

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies,  
    label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show() 

# Evaluate the model on the whole dataset after training
def evaluate_model(model, data_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # No need to compute gradients during evaluation
        for inputs, labels in data_loader:
            # Move data to the selected device
            if device:
                inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs.view(-1, 4), labels.view(-1, 4).argmax(dim=-1))
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, dim=-1)  # Predicted classes
            correct_predictions += (predicted == labels.argmax(dim=-1)).sum().item()
            total_predictions += labels.size(0) * labels.size(1)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct_predictions / total_predictions

    return avg_loss, accuracy