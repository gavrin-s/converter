import os
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, "data")
MODELS_PATH = os.path.join(ROOT_PATH, "models")
