from torchvision.transforms.functional import to_pil_image
from steps.config import ModelConfig
import os

def save_images(img_tensor, folder):
    """
    Saves a batch of tensors as images in a directory.
    """
    config = ModelConfig()
    os.makedirs(folder, exist_ok=True)
    
    img_tensor = img_tensor.reshape(-1, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)  # reshape to [batch, channels, height, width]
    for i, img in enumerate(img_tensor):
        img = to_pil_image(img)
        img.save(os.path.join(folder, f'image_{i}.png'))