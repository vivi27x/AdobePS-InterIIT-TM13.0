from os import listdir
from os.path import isdir, join
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transforms import add_new_channels

class CustomDataset(Dataset):
    """
    Custom dataset class for loading images from a specified directory, transforming them, and returning them in a format suitable for PyTorch models.
    
    Args:
        dataset_path (str): Path to the dataset directory containing image files.
        resolution (int, optional): The resolution to resize images to (default is 224).
        norm_mean (list, optional): Mean values for image normalization (default is IMAGENET_DEFAULT_MEAN).
        norm_std (list, optional): Standard deviation values for image normalization (default is IMAGENET_DEFAULT_STD).
    """

    def __init__(self, dataset_path, resolution=224, norm_mean=IMAGENET_DEFAULT_MEAN, norm_std=IMAGENET_DEFAULT_STD):
        assert isdir(dataset_path), f"got {dataset_path}"
        self.dataset_path = dataset_path
        
        self.items = self.parse_dataset()
        
        assert isinstance(resolution, int) and resolution >= 1, f"got {resolution}"
        self.resolution = resolution
        assert len(norm_mean) == 3
        self.norm_mean = norm_mean
        assert len(norm_std) == 3
        self.norm_std = norm_std

    def parse_dataset(self):
        """
        Parses the dataset directory and returns a list of image paths.
        
        This method checks the provided directory for image files (with extensions .jpg, .png, .jpeg) 
        and returns a list of dictionaries, each containing the image path.
        
        Returns:
            list: A list of dictionaries containing image file paths.
        """
        def is_image(filename):
            for extension in ["jpg", "png", "jpeg"]:
                if filename.lower().endswith(extension):
                    return True
            return False
        
        split_path = self.dataset_path
        items = [{
            "image_path":  join(split_path, image_path),
            } for image_path in listdir(split_path) if is_image(image_path)] 
        return items

    def __len__(self):
        return len(self.items)

    def read_image(self, path):
        image = Image.open(path).convert('RGB')
        image = T.Compose([
            T.Resize(self.resolution + self.resolution // 8, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.resolution),
            T.ToTensor(),
            T.Lambda(add_new_channels),
        ])(image)

        return image
    
    def __getitem__(self, i):
        sample = {
            "image_path": self.items[i]["image_path"],
            "image": self.read_image(self.items[i]["image_path"])
        }
        return sample

