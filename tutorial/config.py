import torch

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HYPERPARAMETERS
INPUT_SIZE = 784
NUM_CLASSES = 10
BATCH_SIZE = 64
NUM_EPOCHS = 3
NUM_WORKERS = 15
