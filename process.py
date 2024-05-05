import numpy as np
import torch
from PIL import Image


def image_to_tensor(image: str) -> torch.Tensor:
    image = Image.open(image)
    image = np.array(image, dtype=np.float32)
    if image.ndim == 3:  # Check if the image has color channels
        image = np.mean(image, axis=2) # Convert to grayscale
    image = image.transpose()
    image = image / 255
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    #image = np.array([image])
    return image