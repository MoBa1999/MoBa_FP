import torch 
import matplotlib.pyplot as plt
import os 
import numpy as np
from Levenshtein import distance

def plot_training_and_test_metrics(output_dir, label = None):
    # Load the saved data
    losses = np.load(os.path.join(output_dir, "training_losses.npy"))
    accuracies = np.load(os.path.join(output_dir, "training_accuracies.npy"))
    test_loss = np.load(os.path.join(output_dir, "test_losses.npy"))
    test_accuracy = np.load(os.path.join(output_dir, "test_accuracies.npy"))
    seqs = np.load(os.path.join(output_dir, "end_seqs.npy"))
    # Plot training accuracy and loss over
    # Training loss plot
    plt.subplot(1, 2, 1)
    #plt.plot(seqs,losses, label="Training Loss", color='blue')
    plt.plot(seqs,accuracies, label="Training Accuracy")
    plt.xlabel("Training Sequences")
    plt.ylabel("Loss / Accuracy")
    plt.title("Training Loss/Accuracy over inputs")
    plt.legend()
    
    # Test loss and accuracy plot
    plt.subplot(1, 2, 2)
    #plt.plot(seqs,test_loss, label="Test Loss", color='blue')
    if label:
        plt.plot(seqs,test_accuracy, label=label)
    else:
        plt.plot(seqs,test_accuracy, label="Test Accuracy")
    plt.xlabel("Training Sequences")
    plt.ylabel("Loss / Accuracy")
    plt.ylim(40,100)
    plt.title("Test Loss/Accuracy over inputs")
    plt.legend()
    
    plt.tight_layout()
    #plt.show()

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
    plt.plot(accuracies, label='Training Accuracy')
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

def evaluate_model_ham(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_hamming_distance = 0
    total_samples = 0

    with torch.no_grad():  # No need to compute gradients during evaluation
        for inputs, labels in data_loader:
            # Move data to the selected device
            if device:
                inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # Forward pass
            # Reshape outputs for decoding and accuracy computation
            #outputs = outputs.permute(0, 2, 1).log_softmax(1)  # Reshape for sequence decoding
            for b in range(outputs.shape[0]):
                # Decode predictions using argmax
                pred_seq = outputs[b,:,:].cpu().detach().numpy()

                # Collapse predictions to remove consecutive duplicates
                pred_seq_collapsed = model.ctc_collapse_probabilities(pred_seq)

                # Decode true labels
                true_seq = labels[b].argmax(dim=-1).cpu().detach().numpy()

                # Compute the Hamming/Levenshtein distance
                total_hamming_distance += distance(pred_seq_collapsed, true_seq)
                total_samples += 1

    avg_hamming_distance = total_hamming_distance / total_samples
    theoretical_accuracy = (model.tar_length - avg_hamming_distance) / model.tar_length * 100

    return theoretical_accuracy

#plot_training_and_test_metrics("/workspaces/MoBa_FP/Experiments/Exp6")
#plot_training_and_test_metrics("/workspaces/MoBa_FP/Experiments/Exp5", label="Single")
plot_training_and_test_metrics("/workspaces/MoBa_FP/Experiments/Exp7", label="Single Simple")
plot_training_and_test_metrics("/workspaces/MoBa_FP/Experiments/Exp8", label="Single Attention")
plot_training_and_test_metrics("/workspaces/MoBa_FP/Experiments/Exp9", label="5 Input Attention")