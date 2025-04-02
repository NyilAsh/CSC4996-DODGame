import torch
from model import PositionPredictor
from data_loader import get_data_loaders
from utils import MaskedBCELoss

def evaluate():
    model = PositionPredictor()
    model.load_state_dict(torch.load('checkpoints/best_model.pt'))
    model.eval()
    
    _, test_loader = get_data_loaders(batch_size=32)
    criterion = MaskedBCELoss()
    
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
    
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')

if __name__ == '__main__':
    evaluate()