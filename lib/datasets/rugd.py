import os
import cv2
import numpy as np
import torch
from .base_dataset import BaseDataset
import albumentations as A


class RUGD(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=3,
                 multi_scale=True,
                 flip=True,
                 ignore_label=255,
                 base_size=2048,
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 augment=False):

        super(RUGD, self).__init__(ignore_label, base_size,
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

        self.color_map = self._get_color_map()
        self.class_weights = torch.FloatTensor([0.5, 5.0]).cuda()

        self.augment = augment
        if self.augment:
            self.albumentations_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomShadow(p=0.15),
            ], p=1.0)

    def _get_color_map(self):
        return np.array([
            [0, 0, 0], [108, 64, 20], [255, 229, 204], [0, 102, 0], [0, 255, 0],
            [0, 153, 153], [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127],
            [64, 64, 64], [255, 128, 0], [255, 0, 0], [153, 76, 0], [102, 102, 0],
            [102, 0, 0], [0, 255, 128], [204, 153, 255], [102, 0, 204],
            [255, 153, 204], [0, 102, 102], [153, 204, 255], [102, 255, 255],
            [101, 101, 11], [114, 85, 47]
        ])

    def _colorize(self, mask_rgb):
        mask_bgr = mask_rgb[:, :, ::-1]
        label = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
        for i, color in enumerate(self.color_map):
            label[np.all(mask_bgr == color, axis=-1)] = i
        return label

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name,
                "weight": 1
            })
        return files

    def remap_mask(self, mask):
        path_classes = [1, 2, 10, 11, 13, 23]
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        for path_class in path_classes:
            new_mask[mask == path_class] = 1
        return new_mask

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image = cv2.imread(os.path.join(self.root, item["img"]), cv2.IMREAD_COLOR)
        label_rgb = cv2.imread(os.path.join(self.root, item["label"]), cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_id = self._colorize(label_rgb)
        label_remapped = self.remap_mask(label_id)

        if self.augment:
            transformed = self.albumentations_transform(image=image, mask=label_remapped)
            image = transformed['image']
            label_remapped = transformed['mask']

        size = image.shape

        image, label = self.gen_sample(image, label_remapped,
                                       self.multi_scale,
                                       self.flip if not self.augment else False)

        return image.copy(), label.copy(), np.array(size), name
