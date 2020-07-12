from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as torch_models
from src.config import DEVICE

TORCHVISION_MODELS = ["mobilenet_v2"]


def get_model(name: str, pretrained: str = "imagenet", checkpoint: Optional[str] = None)\
        -> nn.Module:
    """
    Get torch model by name.
    """

    if checkpoint is not None:
        print(f"Variable `checkpoint` not empty, then set `pretrained` to None")
        pretrained = None

    elif name in TORCHVISION_MODELS:
        model = getattr(torch_models, name)(pretrained=(pretrained == "imagenet"))
    else:
        raise Exception(f"Model {name} not found.")

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)["model_state_dict"])
    model.to(DEVICE)
    return model
