#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os

import torchvision
import torch.utils.data as torch_data
from torchvision import transforms
from utils import BaseEnumOptions


class ImageInterpolation(BaseEnumOptions):
    nearest = transforms.InterpolationMode.NEAREST
    box = transforms.InterpolationMode.BOX
    bilinear = transforms.InterpolationMode.BILINEAR
    hamming = transforms.InterpolationMode.HAMMING
    bicubic = transforms.InterpolationMode.BICUBIC
    lanczos = transforms.InterpolationMode.LANCZOS


class ImageNetDataLoaders(object):
    """
    Data loader provider for ImageNet images, providing a train and a validation loader.
    It assumes that the structure of the images is
        images_dir
            - train
                - label1
                - label2
                - ...
            - val
                - label1
                - label2
                - ...
    """

    def __init__(
        self,
        images_dir: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        interpolation: transforms.InterpolationMode,
    ):
        """
        Parameters
        ----------
        images_dir: str
            Root image directory
        image_size: int
            Number of pixels the image will be re-sized to (square)
        batch_size: int
            Batch size of both the training and validation loaders
        num_workers
            Number of parallel workers loading the images
        interpolation: transforms.InterpolationMode
            Desired interpolation to use for resizing.
        """

        self.images_dir = images_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # For normalization, mean and std dev values are calculated per channel
        # and can be found on the web.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, interpolation=interpolation.value),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(image_size + 24, interpolation=interpolation.value),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self._train_loader = None
        self._val_loader = None

    @property
    def train_loader(self) -> torch_data.DataLoader:
        if not self._train_loader:
            root = os.path.join(self.images_dir, "train")
            train_set = torchvision.datasets.ImageFolder(root, transform=self.train_transforms)
            self._train_loader = torch_data.DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return self._train_loader

    @property
    def val_loader(self) -> torch_data.DataLoader:
        if not self._val_loader:
            root = os.path.join(self.images_dir, "val")
            val_set = torchvision.datasets.ImageFolder(root, transform=self.val_transforms)
            self._val_loader = torch_data.DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return self._val_loader
