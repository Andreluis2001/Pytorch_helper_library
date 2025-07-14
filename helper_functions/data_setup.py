import os
import torch
import requests
import zipfile

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List

def download_data(
    url_source: str,
    destination: str,
    remove_source: bool = True
) -> Path:

  data_dir_path = Path("data")
  data_path = data_dir_path / destination

  if data_path.is_dir():
    print(f"{data_path} already exists, skipping download")

  else:
    data_path.mkdir(parents=True, exist_ok=True)

    target_zip_file = Path(url_source).name

    with open(data_dir_path / target_zip_file, "wb") as f:

      request = requests.get(url_source)
      f.write(request.content)

    with zipfile.ZipFile(data_dir_path / target_zip_file, "r") as zip_f:

      zip_f.extractall(data_path)

    if remove_source:
      os.remove(data_dir_path / target_zip_file)

  return data_path

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transforms: transforms.Compose,
    batch_size: int = 32,
    num_workers: int = 1
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    List[str]
  ]:

  train_data = datasets.ImageFolder(root=train_dir, transform=transforms)
  test_data = datasets.ImageFolder(root=test_dir, transform=transforms)

  class_names = train_data.classes

  train_dataloader = DataLoader(
                                dataset=train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True
                              )
  test_dataloader = DataLoader(
                                dataset=test_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True
                              )

  return train_dataloader, test_dataloader, class_names
