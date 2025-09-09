import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F_trans
import os
import time

import models
from config import config
from config import update_config

CFG_PATH = './experiments/rugd/finetune_ddrnet23.yaml'
MODEL_PATH = './output/finetune_on_binary_data/BinaryMaskDataset/finetune_ddrnet23/best.pth'
IMAGE_PATH = './teste5.png'
OUTPUT_FILENAME = 'segmentation_result'


class PadToSquare:
    def __init__(self, fill_value=0):
        self.fill_value = fill_value

    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        h_padding = (max_dim - w) // 2
        v_padding = (max_dim - h) // 2
        l_pad = h_padding
        r_pad = max_dim - w - l_pad
        t_pad = v_padding
        b_pad = max_dim - h - t_pad
        padding = (l_pad, t_pad, r_pad, b_pad)
        return F_trans.pad(img, padding, self.fill_value, 'constant')


def main():
    class Args:
        cfg = CFG_PATH
        opts = []

    args = Args()
    update_config(config, args)

    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)

    print("Loading weights...")
    pretrained_dict = torch.load(MODEL_PATH, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']

    model_dict = model.state_dict()
    pretrained_dict = {k.replace('model.', ''): v for k, v in pretrained_dict.items() if
                       k.replace('model.', '') in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Weights loaded successfully.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"Model is on device: {device}")

    image = Image.open(IMAGE_PATH).convert('RGB')
    original_image_cv = cv2.imread(IMAGE_PATH)

    transform = transforms.Compose([
        PadToSquare(),
        transforms.Resize((config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    original_tensor = transform(image).unsqueeze(0).to(device)

    flipped_image = F_trans.hflip(image)
    flipped_tensor = transform(flipped_image).unsqueeze(0).to(device)

    print("Running inference with Test-Time Augmentation (TTA)...")
    with torch.no_grad():
        original_output = model(original_tensor)
        if isinstance(original_output, list):
            original_output = original_output[config.TEST.OUTPUT_INDEX]

        flipped_output = model(flipped_tensor)
        if isinstance(flipped_output, list):
            flipped_output = flipped_output[config.TEST.OUTPUT_INDEX]

    flipped_output = torch.flip(flipped_output, dims=[3])

    final_output = (torch.softmax(original_output, dim=1) + torch.softmax(flipped_output, dim=1)) / 2.0

    target_height = config.TEST.IMAGE_SIZE[1]
    target_width = config.TEST.IMAGE_SIZE[0]
    target_size = (target_height, target_width)

    upsampled_output = torch.nn.functional.interpolate(
        final_output,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )
    prediction = torch.argmax(upsampled_output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]

    color_mask_rgb = palette[prediction]

    mask_output_path = f"{OUTPUT_FILENAME}_mask.png"
    overlay_output_path = f"{OUTPUT_FILENAME}_overlay.png"

    cv2.imwrite(mask_output_path, cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR))

    original_image_resized = cv2.resize(original_image_cv, (target_width, target_height))

    overlay = cv2.addWeighted(original_image_resized, 0.6, cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR), 0.4, 0)
    cv2.imwrite(overlay_output_path, overlay)


    print(f"Segmentation mask saved to {mask_output_path} with size {target_width}x{target_height}")
    print(f"Overlay image saved to {overlay_output_path} with size {target_width}x{target_height}")


if __name__ == '__main__':
    main()
