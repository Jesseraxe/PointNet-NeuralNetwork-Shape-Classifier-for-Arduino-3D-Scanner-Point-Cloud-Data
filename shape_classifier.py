# shape_classifier.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PointNet(nn.Module):
    def __init__(self, num_classes=5):
        super(PointNet, self).__init__()
        
        # Input transform net
        self.input_transform = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        
        # Feature transform net
        self.feature_transform = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # Input transform
        x = self.input_transform(x)
        # Global feature
        x = torch.max(x, 2)[0]
        # Dropout for regularization
        x = self.dropout(x)
        # Classification
        x = self.feature_transform(x)
        return x

class PointCloudAugmentor:
    def __init__(
        self,
        jitter_sigma: float = 0.02,
        jitter_clip: float = 0.05,
        scale_low: float = 0.8,
        scale_high: float = 1.25,
        translate_range: float = 0.2
    ):
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def random_jitter(self, points: np.ndarray) -> np.ndarray:
        noise = np.clip(
            np.random.normal(0, self.jitter_sigma, points.shape),
            -self.jitter_clip,
            self.jitter_clip
        )
        return points + noise

    def random_scale(self, points: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.scale_low, self.scale_high)
        return points * scale

    def random_translate(self, points: np.ndarray) -> np.ndarray:
        translation = np.random.uniform(-self.translate_range, self.translate_range, 3)
        return points + translation

    def __call__(self, points: np.ndarray) -> np.ndarray:
        points = self.random_jitter(points.copy())
        points = self.random_scale(points)
        points = self.random_translate(points)
        return points

class ShapeNetDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        num_points: int = 1024,
        augment: bool = True,
        num_augmentations: int = 9  # Augmentations per scan
    ):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.augmentor = PointCloudAugmentor() if augment else None
        self.num_augmentations = num_augmentations if augment else 0
        self.class_folders = ['sphere', 'cone', 'cube', 'cylinder', 'cuboid']
        
        logger.debug(f"Initializing dataset with root_dir: {root_dir}, split: {split}")
        self.setup_data()

    def setup_data(self):
        """Load and prepare the dataset, with detailed logging."""
        data, labels = [], []
        
        # Load data for the current split
        split_dir = os.path.join(self.root_dir, self.split)
        logger.debug(f"Looking for data in: {split_dir}")

        for label, class_name in enumerate(self.class_folders):
            class_path = os.path.join(split_dir, class_name)
            if not os.path.exists(class_path):
                continue

            files = [f for f in os.listdir(class_path) if f.endswith('.TXT')]
            logger.debug(f"Found {len(files)} files in {class_path}")

            for file in files:
                file_path = os.path.join(class_path, file)
                try:
                    points = np.loadtxt(file_path, delimiter=',', skiprows=3)
                    
                    if len(points.shape) != 2 or points.shape[1] != 3:
                        continue
                        
                    # Normalize points
                    centroid = np.mean(points, axis=0)
                    points -= centroid
                    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
                    if dist > 0:
                        points /= dist

                    # Sample points if needed
                    if len(points) > self.num_points:
                        indices = np.random.choice(len(points), self.num_points, replace=False)
                        points = points[indices]
                    elif len(points) < self.num_points:
                        indices = np.random.choice(len(points), self.num_points, replace=True)
                        points = points[indices]
                    
                    # Add original points
                    data.append(points)
                    labels.append(label)
                    
                    # Add augmented versions if in training split
                    if self.split == 'train' and self.augmentor:
                        for _ in range(self.num_augmentations):
                            augmented_points = self.augmentor(points.copy())
                            data.append(augmented_points)
                            labels.append(label)
                            
                        logger.debug(f"Created {self.num_augmentations} augmentations for {file}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue

        if not data:
            raise ValueError(f"No valid data files found in {split_dir}")

        self.data = np.array(data)
        self.labels = np.array(labels)
        
        # Log dataset statistics
        total_original = len(files) if files else 0
        total_augmented = len(self.data) - total_original
        logger.info(f"\nDataset statistics for {self.split} split:")
        logger.info(f"Original samples: {total_original}")
        logger.info(f"Augmented samples: {total_augmented}")
        logger.info(f"Total samples: {len(self.data)}")
        
        # Log per-class statistics
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            logger.info(f"Class {self.class_folders[label]}: {count} samples "
                      f"({count/(1 + self.num_augmentations):.0f} original + "
                      f"{count - count/(1 + self.num_augmentations):.0f} augmented)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        points = self.data[idx]
        label = self.labels[idx]
        return torch.from_numpy(points).float(), label

def train_model(
    data_dir: str,
    num_epochs: int = 100,  # Increased epochs
    batch_size: int = 8,
    learning_rate: float = 0.0005,  # Reduced learning rate
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    logger.info(f"Using device: {device}")
    
    # Load datasets with more augmentation
    train_dataset = ShapeNetDataset(
        data_dir, 
        split='train', 
        augment=True,
        num_augmentations=5  # Increased augmentations
    )
    val_dataset = ShapeNetDataset(data_dir, split='val', augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True  # Prevent issues with last incomplete batch
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate class weights
    class_counts = np.bincount(train_dataset.labels)
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    
    # Initialize model with stronger regularization
    model = PointNet(num_classes=5).to(device)
    
    # Add weight decay to optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01  # L2 regularization
    )
    
    # Modified learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=7,
        verbose=True,
        min_lr=1e-6
    )
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training variables
    best_val_acc = 0.0
    patience = 15  # Increased patience
    patience_counter = 0
    
    # Track metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    logger.info(f"Starting training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (points, labels) in enumerate(train_loader):
            points = points.transpose(2, 1).float().to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for points, labels in val_loader:
                points = points.transpose(2, 1).float().to(device)
                labels = labels.long().to(device)
                
                outputs = model(points)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        logger.info(f'\nEpoch [{epoch+1}/{num_epochs}] Summary:')
        logger.info(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Create assets directory if it doesn't exist
            os.makedirs('assets/model', exist_ok=True)
            
            # Save model with metadata
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_parameters': {
                    'num_epochs': num_epochs,   
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                }
            }, 'assets/model/best_model.pth')
            patience_counter = 0
            logger.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }

def evaluate_model(model_path, data_dir, device='cuda'):
    """
    Evaluate the model on the test set and generate visualization
    """
    # Load model
    model = PointNet(num_classes=5).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test dataset
    test_dataset = ShapeNetDataset(data_dir, split='test', augment=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Evaluation variables
    all_preds = []
    all_labels = []
    test_correct = 0
    test_total = 0

    # Evaluate
    with torch.no_grad():
        for points, labels in test_loader:
            points = points.transpose(2, 1).float().to(device)
            labels = labels.long().to(device)
            
            outputs = model(points)
            _, predicted = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    test_accuracy = 100. * test_correct / test_total

    # Generate classification report
    class_names = ['sphere', 'cone', 'cube', 'cylinder', 'cuboid']
    report = classification_report(all_labels, all_preds, target_names=class_names)

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save results
    os.makedirs('assets/evaluation', exist_ok=True)
    
    # Save confusion matrix plot
    plt.savefig('assets/evaluation/confusion_matrix.png')
    plt.close()
    
    # Save classification report
    with open('assets/evaluation/classification_report.txt', 'w') as f:
        f.write(report)
    
    # Save metrics
    metrics = {
        'test_accuracy': test_accuracy,
        'confusion_matrix': cm.tolist(),
        'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open('assets/evaluation/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    return test_accuracy, report, cm


if __name__ == "__main__":
    import argparse
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    parser = argparse.ArgumentParser(description='Train and evaluate 3D shape classifier')
    parser.add_argument('--mode', type=str, default='both', choices=['train', 'eval', 'both'],
                       help='Mode to run: train, eval, or both')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing the dataset')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                       help='Learning rate for training')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create necessary directories
    os.makedirs('assets/model', exist_ok=True)
    os.makedirs('assets/evaluation', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if args.mode in ['train', 'both']:
        logger.info("Starting training...")
        try:
            params = {
                'data_dir': args.data_dir,
                'num_epochs': args.num_epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate
            }
            model, metrics = train_model(**params)
            logger.info("Training completed successfully!")
            logger.info(f"Best validation accuracy: {metrics['best_val_acc']:.2f}%")
            
            # Save training metrics and plots
            metrics_path = os.path.join('assets', 'training_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'train_losses': metrics['train_losses'],
                    'val_losses': metrics['val_losses'],
                    'train_accs': metrics['train_accs'],
                    'val_accs': metrics['val_accs'],
                    'best_val_acc': metrics['best_val_acc']
                }, f, indent=4)
            
            # Plot and save training curves
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(metrics['train_losses'], label='Train Loss')
            plt.plot(metrics['val_losses'], label='Val Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(metrics['train_accs'], label='Train Acc')
            plt.plot(metrics['val_accs'], label='Val Acc')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('assets/training_curves.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    if args.mode in ['eval', 'both']:
        logger.info("\nStarting evaluation...")
        try:
            # Load the best model
            model_path = 'assets/model/best_model.pth'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            # Load test dataset
            test_dataset = ShapeNetDataset(args.data_dir, split='test', augment=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            # Load model
            model = PointNet(num_classes=5).to(device)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Evaluation variables
            all_preds = []
            all_labels = []
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            criterion = nn.CrossEntropyLoss()
            
            # Evaluate
            logger.info("Running evaluation on test set...")
            with torch.no_grad():
                for points, labels in test_loader:
                    points = points.transpose(2, 1).float().to(device)
                    labels = labels.long().to(device)
                    
                    outputs = model(points)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            test_accuracy = 100. * test_correct / test_total
            test_loss = test_loss / len(test_loader)
            
            # Generate classification report
            class_names = ['sphere', 'cone', 'cube', 'cylinder', 'cuboid']
            report = classification_report(all_labels, all_preds, 
                                        target_names=class_names, 
                                        digits=4)
            
            # Print results
            logger.info("\nTest Results:")
            logger.info(f"Test Loss: {test_loss:.4f}")
            logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
            logger.info("\nClassification Report:")
            logger.info("\n" + report)
            
            # Save classification report
            with open('assets/evaluation/classification_report.txt', 'w') as f:
                f.write(f"Test Loss: {test_loss:.4f}\n")
                f.write(f"Test Accuracy: {test_accuracy:.2f}%\n\n")
                f.write(report)
            
            # Generate and save confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig('assets/evaluation/confusion_matrix.png')
            
            # Generate and save normalized confusion matrix
            plt.figure(figsize=(10, 8))
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names)
            plt.title('Normalized Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig('assets/evaluation/confusion_matrix_normalized.png')
            plt.close()
            
            # Save detailed metrics
            metrics = {
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'confusion_matrix': cm.tolist(),
                'class_names': class_names,
                'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open('assets/evaluation/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info("\nEvaluation completed successfully!")
            logger.info(f"Results saved in assets/evaluation/")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    logger.info("All tasks completed!")