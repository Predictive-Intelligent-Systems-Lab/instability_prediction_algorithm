import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

# Function to parse filenames for type codes
def parse_type_from_filename(filename):
    """Extract the type code from the filename."""
    pattern = r'recurrence_matrix_U0_\d+\.\d+_segment_\d+_type_(\d+)\.npy'
    match = re.match(pattern, filename)
    if match:
        type_code = int(match.group(1))
        # Updated to include all valid type codes (0, 1, 2, 3)
        if type_code in [0, 1, 2, 3]:
            return type_code
    return None

# Custom Dataset for recurrence matrices
class RecurrenceMatrixDataset(Dataset):
    def __init__(self, matrix_dir):
        self.matrix_dir = Path(matrix_dir)
        self.files = list(self.matrix_dir.glob('*.npy'))
        self.type_codes = [parse_type_from_filename(file.name) for file in self.files]
        
        # Filter out files without valid type codes
        valid_indices = [i for i, code in enumerate(self.type_codes) if code is not None]
        self.files = [self.files[i] for i in valid_indices]
        self.type_codes = [self.type_codes[i] for i in valid_indices]
        
        # Print dataset statistics
        type_counts = {}
        for code in self.type_codes:
            type_counts[code] = type_counts.get(code, 0) + 1
            
        print(f"Dataset loaded from {matrix_dir}")
        print(f"Total samples: {len(self.files)}")
        print("Type distribution:")
        for type_code, count in sorted(type_counts.items()):
            type_description = {
                0: "K <= 0.2",
                1: "Type I (2nd quadrant)",
                2: "Type I (4th quadrant)",
                3: "K >= 0.8"
            }.get(type_code, "Unknown")
            print(f"  Type {type_code} ({type_description}): {count} samples ({count/len(self.files):.1%})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        matrix = np.load(self.files[idx])
        type_code = self.type_codes[idx]
        
        # Convert to PyTorch tensors
        matrix = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(type_code, dtype=torch.long)
        return matrix, label

# CNN Model - Updated for 4 classes (0, 1, 2, 3)
class RecurrenceMatrixCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(RecurrenceMatrixCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4),  # 450x450 -> 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112 -> 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 56x56 -> 28x28
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Main execution
if __name__ == "__main__":
    # Get script directory
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Set paths
    input_dir = script_dir / "classified_matrices"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading data from {input_dir}...")
    dataset = RecurrenceMatrixDataset(input_dir)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create model
    model = RecurrenceMatrixCNN(num_classes=4).to(device)  # 4 classes: Types 0, 1, 2, 3
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 50
    
    # For tracking metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Calculate statistics
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                
                # Store predictions and labels for confusion matrix
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct / len(val_dataset)
        val_accuracies.append(val_accuracy)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Every 50 epochs, print per-class accuracy
        if (epoch + 1) % 50 == 0:
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            # Calculate per-class accuracy
            print("\nPer-class accuracy:")
            for class_idx in range(4):
                class_mask = (all_labels == class_idx)
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(all_preds[class_mask] == class_idx)
                    class_desc = {
                        0: "K <= 0.2",
                        1: "Type I (2nd quadrant)",
                        2: "Type I (4th quadrant)",
                        3: "K >= 0.8"
                    }.get(class_idx, f"Type {class_idx}")
                    print(f"  {class_desc}: {class_acc:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), script_dir / "recurrence_matrix_cnn.pth")
    
    # Plot and save training curves
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(script_dir / "training_curves.png", dpi=300)
    
    # Calculate and plot final confusion matrix
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Create confusion matrix
    conf_matrix = np.zeros((4, 4), dtype=int)
    for true_label, pred_label in zip(all_labels, all_preds):
        conf_matrix[true_label, pred_label] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    class_names = ['Type 0\n(K ≤ 0.2)', 'Type 1\n(Type I, 2nd quad)', 'Type 2\n(Type I, 4th quad)', 'Type 3\n(K ≥ 0.8)']
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(script_dir / "confusion_matrix.png", dpi=300)
    
    # Display final results
    print("\nTraining complete!")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    print(f"Model saved to {script_dir / 'recurrence_matrix_cnn.pth'}")
    print(f"Training curves saved to {script_dir / 'training_curves.png'}")
    print(f"Confusion matrix saved to {script_dir / 'confusion_matrix.png'}")
    
    plt.show()