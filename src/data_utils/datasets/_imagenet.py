from pathlib import Path
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
from ._imagenet_classes import IMAGENET2012_CLASSES
import glob
from PIL import Image
import torchvision.transforms as T
from typing import Literal


class ImageNet1k_2012(Dataset):
    """ImageNet 2012 dataset with 1000 classes. Files downloaded from
    https://huggingface.co/datasets/imagenet-1k/blob/main/classes.py
    and unzipped into the dataset directory.

    The dataset directories must have the names 'train_images', 'val_images' and 'test_images'
    and must lie in the root directory.
    """

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "test", "val"],
        transform: Optional[Callable] = T.ToTensor(),
    ):
        """Initializes the dataset.

        Args:
            root: Path to the dataset directory.
            split: The split to load.
            transform: A function/transform that takes in an PIL image and returns a transformed version.
        """
        if isinstance(root, str):
            root = Path(root)
        root = root.expanduser()
        if not root.exists():
            raise ValueError(f"Path {root} does not exist.")

        self.paths = []
        self.labels = []
        self.names = []
        self.transform = transform if transform is not None else lambda x: x

        for path in glob.glob(f"{root}/{split}_images/*.JPEG"):
            self.paths.append(path)
            numerical_name = self.name_from_path(path)
            name, label = IMAGENET2012_CLASSES[numerical_name]
            self.names.append(name)
            self.labels.append(label)

        if len(self.paths) == 0:
            raise ValueError(f"No images found in {root}/{split}_images")

    def show(self, idx: int):
        Image.open(self.paths[idx]).show()

    def name_from_path(self, path: str) -> str:
        return path.split("/")[-1].split("_")[-1].split(".")[0]

    def label_to_name(self, x: int):
        return self.names[x]

    def __getitem__(self, index):
        return (
            self.transform(Image.open(self.paths[index]).convert("RGB")),
            self.labels[index],
        )

    def __len__(self) -> int:
        return len(self.paths)
