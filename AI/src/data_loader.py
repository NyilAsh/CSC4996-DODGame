import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from Config import config

class PositionDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.grid_size = config.GRID_SIZE
        self.input_cols = [...]  # Your input columns
        self.output_cols = [...] # Your output columns

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Initialize grids with -1
        inputs = torch.full((config.INPUT_CHANNELS, config.GRID_SIZE, config.GRID_SIZE), -1.0)
        outputs = torch.full((config.OUTPUT_CHANNELS, config.GRID_SIZE, config.GRID_SIZE), -1.0)
        
        # Fill valid positions (skip -1,-1)
        for i in range(config.INPUT_CHANNELS):
            x, y = int(row[self.input_cols[2*i]]), int(row[self.input_cols[2*i+1]])
            if x >= 0 and y >= 0:
                inputs[i, y, x] = 1.0
                
        for i in range(config.OUTPUT_CHANNELS):
            x, y = int(row[self.output_cols[2*i]]), int(row[self.output_cols[2*i+1]])
            if x >= 0 and y >= 0:
                outputs[i, y, x] = 1.0
                
        return inputs, outputs

def get_data_loaders():
    train_set = PositionDataset(config.TRAIN_CSV)
    test_set = PositionDataset(config.TEST_CSV)
    return (
        DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True),
        DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)
    )