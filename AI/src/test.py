import torch
import numpy as np
from model import PositionPredictor

def test_individual_input(*args, model_path='checkpoints/best_model.pt'):
    """
    Returns predictions in two groups:
    - first_choices: Top prediction for each of the 5 channels
    - second_choices: Second prediction for each of the 5 channels
    """
    # Load model and process input
    model = PositionPredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create input tensor
    input_tensor = torch.full((1, 15, 10, 10), -1.0)
    for i in range(15):
        x, y = args[2*i], args[2*i+1]
        if x != -1 and y != -1:
            input_tensor[0, i, y, x] = 1.0

    # Get predictions
    with torch.no_grad():
        output = model(input_tensor)[0]
        probs = output.sigmoid().numpy()

    # Get top predictions
    results = {'first_choices': [], 'second_choices': []}
    entity_names = ['Attacker A', 'Attacker B', 'Attacker C', 
                   'Defender A', 'Defender B']
    
    for channel_idx in range(5):
        channel_data = probs[channel_idx]
        flat_data = channel_data.flatten()
        top_indices = np.argpartition(flat_data, -2)[-2:]
        sorted_indices = top_indices[np.argsort(-flat_data[top_indices])]
        
        # First choice
        y1, x1 = divmod(sorted_indices[0], 10)
        results['first_choices'].append({
            'entity': entity_names[channel_idx],
            'x': x1,
            'y': y1
        })
        
        # Second choice
        if len(sorted_indices) > 1:
            y2, x2 = divmod(sorted_indices[1], 10)
            results['second_choices'].append({
                'entity': entity_names[channel_idx],
                'x': x2,
                'y': y2
            })

    # Print results
    print("First Choices:")
    for pred in results['first_choices']:
        print(f"{pred['entity']}: ({pred['x']}, {pred['y']})")
    
    print("\nSecond Choices:")
    for pred in results['second_choices']:
        print(f"{pred['entity']}: ({pred['x']}, {pred['y']})")

    return results

# Example usage
if __name__ == '__main__':
    test_individual_input(
        2,8,2,9,3,9
        ,5,6,4,7,4,9
        ,5,7,5,8,5,9
        ,2,7,0,9,9,4
        ,6,4,3,5,2,8  # Defender B
    )