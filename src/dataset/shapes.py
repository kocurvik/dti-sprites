from torch.utils.data import Dataset
import os.path as osp
from torchvision import transforms
from skimage import io
import os
import numpy as np
import torch
from PIL import Image, ImageFile
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

def add_random_shape(img, max_diam=32, max_vertices=10):
    center = np.random.rand(2) * img.shape[:2]

    diam = np.random.rand() * max_diam / 2 + max_diam
    angle = np.random.rand() * np.pi * 2

    n_vertices = np.random.randint(3, max_vertices)

    vertices = np.array([center + diam * np.array([np.cos(angle + i * np.pi * 2 / n_vertices), np.sin(angle + i * np.pi * 2/n_vertices)]) for i in range(n_vertices)], dtype=np.int32)

    random_color = np.random.rand(3).tolist()

    # img = cv2.polylines(img, [vertices], True, random_color, thickness=0)
    img = cv2.fillConvexPoly(img, np.array([vertices]), random_color)
    return img



class ShapesDataset(Dataset):
    def __init__(self, split, img_size=(128, 128), max_diam=10, size=1200, max_vertices=10):
        self.img_size = img_size
        self.n_channels = 3
        self.max_diam = max_diam

        if split == 'test' or split == 'val':
            self.size = 120
        else:
            self.size = size

        self.max_vertices = max_vertices
        self.n_classes = max_vertices - 3



    def __getitem__(self, index):
        default_bg_gray = np.random.rand()
        bg_noise = np.random.rand(self.img_size[0], self.img_size[1], 1) * 0.05
        img = np.ones([self.img_size[0], self.img_size[1], 3], dtype=np.float32) * default_bg_gray
        img += bg_noise

        n_triangles = np.random.randint(2, 10)

        for i in range(n_triangles):
            img = add_random_shape(img, max_diam=self.max_diam, max_vertices=self.max_vertices)

        tensor_img =  transforms.ToTensor()(img)
        return tensor_img, torch.zeros_like(tensor_img, dtype=torch.long)

    def __len__(self):
        # return len(self.img_paths)
        return self.size


