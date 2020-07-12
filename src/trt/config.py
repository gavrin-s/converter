import os
from src.config import MODELS_PATH, DATA_PATH
from src.models import get_model


GET_MODEL_FUNCTION = get_model  # Function return Torch model
MODEL_NAME = "mobilenet_v2"
BATCH_SIZE = 1
IMAGE_SIZE = (224, 224)
CHECKPOINT = None  # os.path.join(MODELS_PATH, "mobilenet_v2.pth")
SIMPLIFYING = True
ONNX_MODEL_FILE = os.path.join(MODELS_PATH, f"{MODEL_NAME}.onnx")
TRT_MODEL_ENGINE = os.path.join(MODELS_PATH, f"{MODEL_NAME}.engine")

IMAGE_PATH = None  # os.path.join(DATA_PATH, "cat.jpg")
