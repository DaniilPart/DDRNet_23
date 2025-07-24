import os
import cv2
import numpy as np
import torch
from .base_dataset import BaseDataset
import albumentations as A

class BinaryMaskDataset(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=2,
                 multi_scale=True,
                 flip=True,
                 ignore_label=255,
                 base_size=2048,
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 augment=True):

        super(BinaryMaskDataset, self).__init__(ignore_label, base_size,
                                         crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [line.strip().split() for line in open(os.path.join(self.root, self.list_path))]
        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.class_weights = torch.FloatTensor([0.5, 5.0]).cuda()

        self.augment = augment
        if self.augment:
            self.albumentations_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3),
                A.RandomBrightnessContrast(p=0.2),
            ], p=1.0)

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })
        return files

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image = cv2.imread(os.path.join(self.root, item["img"]), cv2.IMREAD_COLOR)
        label = cv2.imread(os.path.join(self.root, item["label"]), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found at {os.path.join(self.root, item['img'])}")
        if label is None:
            raise FileNotFoundError(f"Label not found at {os.path.join(self.root, item['label'])}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augment:
            transformed = self.albumentations_transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']

        size = image.shape

        image, label = self.gen_sample(image, label,
                                       self.multi_scale,
                                       self.flip if not self.augment else False)

        return image.copy(), label.copy(), np.array(size), name
