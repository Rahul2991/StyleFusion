from sklearn.preprocessing import MultiLabelBinarizer
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ArtDataset(Dataset):
    def __init__(self, dataset: list, transform: transforms.transforms.Compose = None):
        """
        Args:
            dataset: list
            transform: torchvision.transforms.transforms.Compose (optional)
        """
        self.dataset = dataset
        self.mlb = MultiLabelBinarizer()
        genres = [data['genre'] for data in self.dataset]
        self.mlb.fit(genres)
        self.transform = transform if transform else transforms.ToTensor()
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.dataset[idx]
        image_path = data['img_loc']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        genre_vector = self.mlb.transform([data['genre']])[0]

        return image, torch.tensor(genre_vector, dtype=torch.float32)