import os
import matplotlib.pyplot as plt


def plot_progress(split_name, losses, accuracies, f1_scores, output_dir):
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(18, 6))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses, label=f"{split_name} Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{split_name} Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, accuracies, label=f"{split_name} Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{split_name} Accuracy per Epoch')

    # Plot f1 score
    plt.subplot(1, 3, 3)
    plt.plot(epochs, f1_scores, label=f"{split_name} F1 Score", color='red')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'{split_name} F1 Score per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{split_name}_progress.png"))
    plt.close()


def plot_all(train_losses, train_accuracies, train_f1s, val_losses, val_accuracies, val_f1s, output_dir):
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(18, 6))

    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label=f"Train Loss", color='red')
    plt.plot(epochs, val_losses, label=f"Val Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss per Epoch')

    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, label=f"Train Accuracy", color='red')
    plt.plot(epochs, val_accuracies, label=f"Val Accuracy", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy per Epoch')

    # Plot f1 scores
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_f1s, label=f"Train F1 Score", color='red')
    plt.plot(epochs, val_f1s, label=f"Val F1 Score", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"full_plot.png"))
    plt.close()