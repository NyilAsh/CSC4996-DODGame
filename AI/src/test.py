import torch
from model import PositionPredictor
from Config import config

def test_individual_input(*args, model_path=None, threshold=0.5):
    """Test model with individual coordinates"""
    model = PositionPredictor()
    model.load_state_dict(torch.load(model_path or config.CHECKPOINT_PATH))
    model.eval()
    
    # Create input tensor
    input_tensor = torch.full((1, config.INPUT_CHANNELS, config.GRID_SIZE, config.GRID_SIZE), -1.0)
    for i in range(15):
        x, y = args[2*i], args[2*i+1]
        if x != -1 and y != -1:
            input_tensor[0, i, y, x] = 1.0
    
    # Get prediction
    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor))
    
    # Process output
    predictions = []
    for i in range(5):
        channel = output[0, i]
        max_val, max_idx = channel.max(), channel.argmax()
        if max_val > threshold:
            y, x = max_idx // config.GRID_SIZE, max_idx % config.GRID_SIZE
            predictions.append((x.item(), y.item(), max_val.item()))
        else:
            predictions.append(None)
    
    return predictions

# Example usage
if __name__ == '__main__':
    # Replace with actual coordinates
    preds = test_individual_input(*[-1]*30)  
    for i, pred in enumerate(preds):
        print(f"Entity {i}: {pred if pred else 'No prediction'}")