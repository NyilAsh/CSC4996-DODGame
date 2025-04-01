import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class PositionDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.grid_size = 10
        
        # Input columns (previous positions)
        self.input_columns = [
            'attacker A\'s prev1 x', 'attacker A\'s prev1 y',
            'attacker A\'s prev2 x', 'attacker A\'s prev2 y',
            'attacker A\'s prev3 x', 'attacker A\'s prev3 y',
            'attacker B\'s prev1 x', 'attacker B\'s prev1 y',
            'attacker B\'s prev2 x', 'attacker B\'s prev2 y',
            'attacker B\'s prev3 x', 'attacker B\'s prev3 y',
            'attacker C\'s prev1 x', 'attacker C\'s prev1 y',
            'attacker C\'s prev2 x', 'attacker C\'s prev2 y',
            'attacker C\'s prev3 x', 'attacker C\'s prev3 y',
            'defender A\'s prev1 x', 'defender A\'s prev1 y',
            'defender A\'s prev2 x', 'defender A\'s prev2 y',
            'defender A\'s prev3 x', 'defender A\'s prev3 y',
            'defender B\'s prev1 x', 'defender B\'s prev1 y',
            'defender B\'s prev2 x', 'defender B\'s prev2 y',
            'defender B\'s prev3 x', 'defender B\'s prev3 y'
        ]
        
        # Output columns (current positions)
        self.output_columns = [
            'attacker A current x', 'attacker A current y',
            'attacker B current x', 'attacker B current y',
            'attacker C current x', 'attacker C current y',
            'defender A current x', 'defender A current y',
            'defender B current x', 'defender B current y'
        ]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Input images (15, 10, 10) - initialized to -1
        input_images = torch.full((15, self.grid_size, self.grid_size), -1.0)
        
        # Output images (5, 10, 10) - initialized to -1
        output_images = torch.full((5, self.grid_size, self.grid_size), -1.0)
        
        # Fill tensors (only set valid positions)
        for i in range(15):
            x, y = int(row[self.input_columns[2*i]]), int(row[self.input_columns[2*i+1]])
            if x >= 0 and y >= 0:
                input_images[i, y, x] = 1.0
                
        for i in range(5):
            x, y = int(row[self.output_columns[2*i]]), int(row[self.output_columns[2*i+1]])
            if x >= 0 and y >= 0:
                output_images[i, y, x] = 1.0
        
        return input_images, output_images

def get_data_loaders(batch_size=32):
    train_dataset = PositionDataset('data\data.csv')
    test_dataset = PositionDataset('data\data.csv')
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )