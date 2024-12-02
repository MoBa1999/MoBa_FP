import torch 
import matplotlib.pyplot as plt
import os 
import numpy as np
from Levenshtein import distance
from PIL import Image

def plot_training_and_test_metrics(output_dir, label = None):
    # Load the saved data
    #losses = np.load(os.path.join(output_dir, "training_losses.npy"))
    accuracies = np.load(os.path.join(output_dir, "training_accuracies.npy"))
    #test_loss = np.load(os.path.join(output_dir, "test_losses.npy"))
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


def plot_training_curves_separate(losses, n_accuracies, ham_accuracies, test_accs, save_path = None):
  """
  Plots training losses, n-accuracies, Ham-accuracies, and test accuracies on separate plots.

  Args:
    losses: List of training losses for each epoch.
    n_accuracies: List of n-accuracies for each epoch (optional).
    ham_accuracies: List of Ham-accuracies for each epoch.
    test_accs: List of test accuracies for each epoch (if available).
    epochs: Number of training epochs.
  """

  plt.figure(figsize=(12, 6))

  # Plot training losses
  plt.subplot(2, 2, 1)
  plt.plot(losses)
  plt.xlabel('Epoch')
  plt.ylabel('Training Loss')
  plt.title('Training Loss')

  # Plot n-accuracies (optional, depending on definition)
  if n_accuracies:
    plt.subplot(2, 2, 2)
    plt.plot(n_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('N-Accuracy')
    plt.title('N-Accuracy')


  # Plot Ham-accuracies
  plt.subplot(2, 2, 3)
  plt.plot(ham_accuracies, label = "Train Lev Accuracy")
  if test_accs:
      plt.plot(test_accs, label='Test Lev Accuracy', color='orange')
  plt.xlabel('Epoch')
  plt.ylabel('Lev-Accuracy')
  plt.title('Lev-Accuracy')
  plt.legend()
  plt.tight_layout()
  plt.show()
  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved at {save_path}")
# Evaluate the model on the whole dataset after training
def evaluate_model(model, data_loader, criterion, device, tar_len=200):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    ham_loss = 0
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

            for b in range(predicted.shape[0]):
                ham_loss+= distance(predicted[b,:].cpu().detach().numpy(),labels.argmax(dim=-1)[b,:].cpu().detach().numpy())

    avg_ham = ham_loss/ (total_predictions/tar_len)
    ham_ac = (tar_len - avg_ham)/tar_len * 100
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct_predictions / total_predictions

    return avg_loss, accuracy, ham_ac

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
                true_seq = labels[b,:].argmax(dim=-1).cpu().detach().numpy() +1

                # Compute the Hamming/Levenshtein distance
                total_hamming_distance += distance(pred_seq_collapsed, true_seq)
                total_samples += 1

    avg_hamming_distance = total_hamming_distance / total_samples
    theoretical_accuracy = (model.tar_length - avg_hamming_distance) / model.tar_length * 100

    return theoretical_accuracy

def show_im(path):
    image = Image.open(path)
    image.show()



#plot_training_and_test_metrics("/workspaces/MoBa_FP/Experiments/Exp6")
#plot_training_and_test_metrics("/workspaces/MoBa_FP/Experiments/Exp5", label="Single")
#plot_training_and_test_metrics("/workspaces/MoBa_FP/Experiments/Exp7", label="Single Simple")
#plot_training_and_test_metrics("/workspaces/MoBa_FP/Experiments/Exp8", label="Single Attention")
#plot_training_and_test_metrics("/workspaces/MoBa_FP/Experiments/Exp9", label="5 Input Attention")
#plot_training_and_test_metrics("/workspaces/MoBa_FP/Experiments/Exp10", label="1 Input CTC Loss")
#plot_training_and_test_metrics("/workspaces/MoBa_FP/Experiments/Exp_real_1", label="Real Data - 1 Input CTC Loss")
#show_im("/media/hdd1/MoritzBa/Plots/10000_s_75_ep.png")
#show_im("/media/hdd1/MoritzBa/Plots/40000_s_75_ep_1_r.png")
show_im("/media/hdd1/MoritzBa/Plots/CTC_70000_s_100_ep_1_r.png")