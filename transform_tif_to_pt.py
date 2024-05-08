from osgeo import gdal
import numpy
import torch
import torchvision as vision
import tqdm
from typing import Callable, Tuple, List, Any, Union
from pathlib import Path
import os

has_cuda = torch.cuda.is_available()

def geo_tiff_loader(path: str):
    path = Path(path)
    data = gdal.Open(path)
    data = data.ReadAsArray()
    return path.stem, data


class GeoTiffImageFolder(vision.datasets.DatasetFolder):
    def __init__(self, root: Union[str, Path], transform: Callable = None, target_transform: Callable = None):
        super().__init__(
            root=root,
            loader=geo_tiff_loader,
            extensions=[".tif", ".tiff"],
            transform=transform,
            target_transform=target_transform,
            is_valid_file=None,
            allow_empty=True,
        )


dataset = GeoTiffImageFolder(root="dataset13")

print("Creating class directories...")

for label in dataset.classes:
    os.makedirs("dataset13_pt/" + label, exist_ok=True)

print("Converting .tif files...")

for (image_file, data), label in tqdm.tqdm(dataset):
    label = dataset.classes[label]
    with open("dataset13_pt/" + label + "/" + image_file + ".pt", mode="wb") as fp:
        torch.save(data, fp)
