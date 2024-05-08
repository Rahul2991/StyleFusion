from torch.utils.data import DataLoader, random_split
from module.data_loader import ArtDataset
from torchvision import transforms
from .config import ModelConfig
from typing_extensions import Annotated
from typing import Tuple
from zenml import step
import logging, sys

@step
def preprocess_data(
    dataset: list,
    config: ModelConfig
    ) -> Tuple[Annotated[DataLoader, "train_dataloader"], Annotated[DataLoader, "val_dataloader"], Annotated[DataLoader, "test_dataloader"]]:
    """A `step` to load the Artists dataset."""

    try:
        logging.info('Preparing to preprocess dataset')
        
        transform = transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 360)),
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor()
        ])
        
        dataset = ArtDataset(dataset=dataset, transform=transform)
        
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * train_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE)
        test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

        logging.info('Dataset preprocessing successful')
        return train_dataloader, val_dataloader, test_dataloader
    except Exception as e:
        _, _, line = sys.exc_info()
        logging.error(f"Error in Data preprocessing: {e}. Line No.{line.tb_lineno}")
        raise e