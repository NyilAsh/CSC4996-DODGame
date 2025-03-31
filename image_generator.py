import cv2
import numpy as np
import base64

def create_game_state_image(board, attackers, defenders, shot_tiles=None, show_paths=False):
    """
    Create a visualization of the game state using OpenCV.
    
    Args:
        board: 2D array representing the game board
        attackers: List of attacker objects with positions and paths
        defenders: List of defender positions
        shot_tiles: List of tiles that have been targeted
        show_paths: Boolean to toggle attacker path visualization
    
    Returns:
        numpy.ndarray: Image of the game state
    """
    img = np.ones((600, 600, 3), dtype=np.uint8) * 255
    
    GRID_SIZE = 10
    CELL_SIZE = 50
    
    cv2.rectangle(img, (25, 20), (525, 520), (240, 240, 240), -1)
    
    for i in range(GRID_SIZE + 1):
        cv2.line(img, (i * CELL_SIZE + 25, 20), (i * CELL_SIZE + 25, 520), (200, 200, 200), 1)
        cv2.line(img, (25, i * CELL_SIZE + 20), (525, i * CELL_SIZE + 20), (200, 200, 200), 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(GRID_SIZE):
        cv2.putText(img, chr(65 + i), (i * CELL_SIZE + 50, 15), font, 0.5, (100, 100, 100), 1)
        cv2.putText(img, str(i + 1), (5, i * CELL_SIZE + 45), font, 0.5, (100, 100, 100), 1)
    
    if shot_tiles:
        for tile in shot_tiles:
            r, c = tile
            overlay = img.copy()
            cv2.rectangle(overlay, 
                         (c * CELL_SIZE + 25, r * CELL_SIZE + 20),
                         ((c + 1) * CELL_SIZE + 25, (r + 1) * CELL_SIZE + 20),
                         (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.rectangle(img, 
                         (c * CELL_SIZE + 25, r * CELL_SIZE + 20),
                         ((c + 1) * CELL_SIZE + 25, (r + 1) * CELL_SIZE + 20),
                         (0, 0, 255), 2)
    
    for defender in defenders:
        r, c = defender
        cv2.rectangle(img,
                     (c * CELL_SIZE + 30, r * CELL_SIZE + 25),
                     ((c + 1) * CELL_SIZE + 20, (r + 1) * CELL_SIZE + 15),
                     (255, 150, 0), -1)  
        cv2.rectangle(img,
                     (c * CELL_SIZE + 30, r * CELL_SIZE + 25),
                     ((c + 1) * CELL_SIZE + 20, (r + 1) * CELL_SIZE + 15),
                     (255, 100, 0), 2)  
    
    for attacker in attackers:
        if show_paths:
            path = attacker['steppedPath']
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                progress = i / (len(path) - 1)
                color = (
                    int(0 * (1 - progress) + 255 * progress),  
                    int(255 * (1 - progress) + 0 * progress),  
                    int(0 * (1 - progress) + 0 * progress)     
                )
                cv2.line(img,
                        (start[1] * CELL_SIZE + 50, start[0] * CELL_SIZE + 30),
                        (end[1] * CELL_SIZE + 50, end[0] * CELL_SIZE + 30),
                        color, 2)
        
        current_pos = attacker['steppedPath'][attacker['currentIndex']]
        r, c = current_pos
        
        points = np.array([
            [c * CELL_SIZE + 45, r * CELL_SIZE + 25],  
            [c * CELL_SIZE + 60, r * CELL_SIZE + 35],  
            [c * CELL_SIZE + 45, r * CELL_SIZE + 45],  
            [c * CELL_SIZE + 30, r * CELL_SIZE + 35]   
        ], np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [points], (0, 0, 255))
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        cv2.polylines(img, [points], True, (0, 0, 200), 2)
    
    return img

def generate_state_images(current_board, attackers, defenders, shot_tiles, show_paths=False):
    """
    Generate images for current and predicted game states.
    
    Args:
        current_board: Current game board state
        attackers: List of current attackers
        defenders: List of current defenders
        shot_tiles: List of current shot tiles
        show_paths: Boolean to toggle attacker path visualization
    
    Returns:
        dict: Dictionary containing base64 encoded images for different states
    """
    images = {}
    
    # Generate current state
    current_state = create_game_state_image(current_board, attackers, defenders, shot_tiles, show_paths)
    images['current'] = encode_image_to_base64(current_state)
    
    # Generate predictions for next 3 turns
    for i in range(1, 4):
        # Create a copy of the current state for prediction
        pred_board = [row[:] for row in current_board]
        pred_attackers = []
        
        # Update attacker positions based on their paths
        for attacker in attackers:
            if attacker['currentIndex'] + i < len(attacker['steppedPath']):
                pred_attackers.append({
                    'steppedPath': attacker['steppedPath'],
                    'currentIndex': attacker['currentIndex'] + i
                })
        
        # Generate prediction image
        pred_state = create_game_state_image(pred_board, pred_attackers, defenders, shot_tiles, show_paths)
        images[f'prediction_{i}'] = encode_image_to_base64(pred_state)
    
    return images

def encode_image_to_base64(image):
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8') 