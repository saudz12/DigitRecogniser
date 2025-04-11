# Model parameters
INPUT_SIZE = 784  # 28x28 pixels
NUM_CLASSES = 10  # digits 0-9

# Training parameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.5
BATCH_SIZE = 128
EPOCHS = 100

# Paths
TRAIN_DATA_PATH = './data/mnist_train.csv'
TEST_DATA_PATH = './data/mnist_test.csv'
MODEL_SAVE_PATH = 'mnist_model.pth'

# GUI parameters
CANVAS_SIZE = 280  # 10x the MNIST image size for easier drawing
MNIST_SIZE = 28    # Original MNIST image size
LINE_WIDTH = 15    # Drawing line width