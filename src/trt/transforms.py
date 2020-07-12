from typing import Tuple, List, Callable
import numpy as np
import cv2
import torch

from src.trt.config import IMAGE_SIZE
from src.config import DEVICE


class Resize:
    """
    Resize transformer. This resizes the image to the required size.
    """
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image: np.ndarray) -> np.ndarray:
        new_image = cv2.resize(image.copy(), (self.height, self.width), interpolation=cv2.INTER_LINEAR)
        return new_image


class Normilize:
    """
    Normalize transformer. Apply standard normalization to image, default values from ImageNet.
    """
    def __init__(self, mean: Tuple[float] = (0.485, 0.456, 0.406), std: Tuple[float] = (0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32) * 255.0
        self.std = np.array(std, dtype=np.float32) * 255.0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        denominator = np.reciprocal(self.std, dtype=np.float32)
        new_image = image.astype(np.float32)
        new_image -= self.mean
        new_image *= denominator
        return new_image


class ToTensor:
    """
    To Tensor transformer. Converts numpy-image array to torch-image tensor: transposes axes, numpy -> torch, to device.
    """
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image.transpose((2, 0, 1))).to(DEVICE)


class Compose:
    """
    Compose transformer. Composes several transformers to one.
    """
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            image = transform(image)
        return image


TRANSFORMS = Compose([
    Resize(*IMAGE_SIZE),
    Normilize()
])

TORCH_TRANSFORMS = Compose([TRANSFORMS, ToTensor()])


def torch_preprocessing(image: np.ndarray) -> torch.Tensor:
    """
    Convert numpy-image array for inference Torch model.
    """
    return TORCH_TRANSFORMS(image)[None]


def onnx_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Convert numpy-image array for inference ONNX model.
    """
    image = TRANSFORMS(image)
    image = image.transpose((2, 0, 1))[None]
    return image


def trt_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Convert numpy-image array for inference TensorRT model.
    """
    image = TRANSFORMS(image)
    image = np.array(image.transpose((2, 0, 1))[None], dtype=np.float32, order="C")
    return image
