import torch
from model import PositionPredictor
from data_loader import PositionDataset

def test_individual_input(*args, model_path='checkpoints/best_model.pt', visualize=True):
    """Test with individual coordinates (-1,-1 for empty positions)"""
    # Load model
    model = PositionPredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Validate input
    if len(args) != 30:
        raise ValueError(f"Expected 30 coordinates, got {len(args)}")
    
    # Create input tensor (15,10,10)
    input_tensor = torch.full((1, 15, 10, 10), -1.0)
    for i in range(15):
        x, y = args[2*i], args[2*i+1]
        if x != -1 and y != -1:
            input_tensor[0, i, y, x] = 1.0
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)
    
    # Process output
    predictions = []
    for i in range(5):
        channel = probs[0, i]
        max_idx = channel.argmax()
        y, x = max_idx // 10, max_idx % 10
        confidence = channel[y, x].item()
        predictions.append({
            'entity': ['Attacker A', 'Attacker B', 'Attacker C', 'Defender A', 'Defender B'][i],
            'x': x,
            'y': y,
            'confidence': confidence
        })
    
    # Print results
    print("\nPredictions:")
    for pred in predictions:
        print(f"{pred['entity']}: ({pred['x']}, {pred['y']}) conf={pred['confidence']:.2f}")
    
    return predictions

# Example usage
if __name__ == '__main__':
    test_individual_input(
        2, 9,-1, -1,-1, -1,
        4, 9, -1, -1, -1, -1,
        8, 9, -1, -1, -1, -1
        , 1, 9, -1, -1, -1, -1,
        9,9,-1,-1,-1,-1  # Defender B
    )