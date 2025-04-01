import torch
import os
from model import PositionPredictor
from data_loader import get_data_loaders
from utils import EarlyStopping, MaskedBCELoss

def train_model():
    # Training parameters
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    patience = 5
    checkpoint_path = '../checkpoints/best_model.pt'
    
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
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Early stopping check
        early_stopping(val_loss/len(val_loader), model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    print("Training complete")

if __name__ == '__main__':
    os.makedirs('../checkpoints', exist_ok=True)
    train_model()