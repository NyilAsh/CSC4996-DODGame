import torch
import os
import numpy as np
from model import PositionPredictor
from data_loader import get_data_loaders
from utils import EarlyStopping, MaskedBCELoss

def calculate_accuracy(outputs, targets, threshold=0.5):
    """Calculate percentage of correct predictions"""
    with torch.no_grad():
        # Convert outputs to predicted positions
        probs = torch.sigmoid(outputs)
        batch_size, num_entities, h, w = probs.shape
        
        # Get predicted positions
        pred_positions = torch.zeros_like(targets)
        for b in range(batch_size):
            for e in range(num_entities):
                # Only consider predictions above threshold
                if probs[b,e].max() >= threshold:
                    max_idx = probs[b,e].argmax()
                    y, x = max_idx // w, max_idx % w
                    pred_positions[b,e,y,x] = 1
        
        # Compare with targets (only count non-empty positions)
        correct = 0
        total = 0
        mask = (targets != -1)
        correct = ((pred_positions == targets) & mask).sum().item()
        total = mask.sum().item()
        
        return correct / total if total > 0 else 0.0

def train_model():
    # Training parameters
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    patience = 5
    checkpoint_path = 'checkpoints/best_model.pt'
    accuracy_threshold = 0.5  # Confidence threshold for considering a prediction valid
    
    # Initialize
    model = PositionPredictor()
    criterion = MaskedBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader, val_loader = get_data_loaders(batch_size)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, path=checkpoint_path)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += calculate_accuracy(outputs, targets, accuracy_threshold) * targets.size(0)
            train_total += targets.size(0)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                val_correct += calculate_accuracy(outputs, targets, accuracy_threshold) * targets.size(0)
                val_total += targets.size(0)
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {avg_train_loss:.4f} | Accuracy: {train_acc:.2%}')
        print(f'Val Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.2%}')
        
        # Early stopping check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    print("\nTraining complete")
    print(f"Best validation accuracy: {early_stopping.best_score:.2%}")

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    train_model()