import torch
from model import PositionPredictor
from data_loader import get_data_loaders
from utils import MaskedBCELoss
from Config import config

def evaluate():
    model = PositionPredictor()
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH))
    model.eval()
    
    _, test_loader = get_data_loaders()
    criterion = MaskedBCELoss()
    
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            test_loss += criterion(model(inputs), targets).item()
    
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')

if __name__ == '__main__':
    evaluate()