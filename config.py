
# Training Configuration
IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 46

# Paths
TRAIN_DIR = '/workspaces/Leaf_Disease_Detection/dataset/train'
VAL_DIR = '/workspaces/Leaf_Disease_Detection/dataset/val'
TEST_DIR = '/workspaces/Leaf_Disease_Detection/dataset/test'

# Model
BASE_MODEL = 'EfficientNetB3'  # Options: EfficientNetB0, EfficientNetB3, MobileNetV2

# Callbacks
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 3
