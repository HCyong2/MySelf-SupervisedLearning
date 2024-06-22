"""
Rotation_test -

Author:霍畅
Date:2024/6/14
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import random


def rotate_image(img, angle):
    if angle == 0:
        return img
    elif angle == 1:
        return img.transpose(Image.ROTATE_90)
    elif angle == 2:
        return img.transpose(Image.ROTATE_180)
    elif angle == 3:
        return img.transpose(Image.ROTATE_270)
    else:
        raise ValueError("Angle must be 0, 90, 180, or 270 degrees")


angle_list = [0, 90, 180, 270]
img = Image.open("plots/test_image.jpg")
rotated_image_list = []
for i in range(4):
    rotated_image = rotate_image(img, i)
    rotated_image_list.append(rotated_image)

fig, axes = plt.subplots(1, 4, figsize=(8, 5))
for i in range(4):
    axes[i].imshow(rotated_image_list[i])
    axes[i].set_title(f'Rotate {angle_list[i]}°')
    axes[i].axis('off')
plt.suptitle(f'Image Rotation Example', fontsize=16)
plt.savefig("./plots/image_rotation_example.jpg")
plt.show()
