import torch
import os
from model import PositionPredictor
from data_loader import get_data_loaders
from utils import EarlyStopping, MaskedBCELoss
from Config import config

def train_model():
    model = PositionPredictor()
    criterion = MaskedBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    train_loader, val_loader = get_data_loaders()
    
    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        delta=config.DELTA,
        path=config.CHECKPOINT_PATH
    )
    
    for epoch in range(config.EPOCHS):
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
                val_loss += criterion(model(inputs), targets).item()
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        
        early_stopping(val_loss/len(val_loader), model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    print("Training complete")

if __name__ == '__main__':
    os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)
    train_model()