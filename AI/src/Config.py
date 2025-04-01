# Configuration parameters
class config:
    # Data
    GRID_SIZE = 10
    INPUT_CHANNELS = 15  # 5 entities × 3 time steps
    OUTPUT_CHANNELS = 5   # 5 current positions
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 100
    PATIENCE = 5          # Early stopping patience
    DELTA = 0.001        # Minimum improvement threshold
    
    # Paths
    TRAIN_CSV = 'data/data.csv'
    TEST_CSV = 'data/testdata.csv'
    CHECKPOINT_PATH = '../checkpoints/best_model.pt'
    
config = config()