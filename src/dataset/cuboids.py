from torch.utils.data import Dataset
import os.path as osp
from torchvision import transforms
from skimage import io
import os
import numpy as np
import torch
from PIL import Image, ImageFile
import cv2

from utils.path import DATASETS_PATH

ImageFile.LOAD_TRUNCATED_IMAGES = True


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background

    code by aaronsnoswell from https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image = np.array(image)
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(np.floor(angle / (np.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else np.pi - angle
    alpha = (sign_alpha % np.pi + np.pi) % np.pi

    bb_w = w * np.cos(alpha) + h * np.sin(alpha)
    bb_h = w * np.sin(alpha) + h * np.cos(alpha)

    gamma = np.arctan2(bb_w, bb_w) if (w < h) else np.arctan2(bb_w, bb_w)

    delta = np.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * np.cos(alpha)
    a = d * np.sin(alpha) / np.sin(delta)

    y = a * np.cos(gamma)
    x = y * np.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def randomly_rotate_image(image):
    angle = np.random.rand() * 360
    image_rotated = rotate_image(image, angle)

    image_height, image_width = image.shape[0:2]

    image_rotated_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            np.radians(angle)
        )
    )

    return image_rotated_cropped


class Cuboids(Dataset):
    def __init__(self, split, img_size=(128, 128), size=60000, **kwargs):
        # checkpointdir = os.checkpointdir.join(root, mode)
        assert split in ['train', 'val', 'test']
        self.root = os.path.join(DATASETS_PATH, 'cuboids')
        self.mode = split
        self.img_size = img_size
        self.size = size if self.mode == 'train' else 96
        self.n_classes = 1
        self.n_channels = 3

        assert os.path.exists(self.root), 'Path {} does not exist'.format(self.root)

        self.eval_mode = kwargs.get('eval_mode', False) or split == 'test'
        self.eval_semantic = kwargs.get('eval_semantic', False)
        self.instance_eval = True

        self.img_paths = []
        # img_dir = os.path.join(self.root, mode)
        img_dir = self.root
        for file in os.scandir(img_dir):
            img_path = file.path
            if 'png' in img_path or 'jpg' in img_path:
                self.img_paths.append(img_path)

        # get_index = lambda x: int(os.path.basename(x).split('-')[0])
        # self.img_paths.sort(key=get_index)

    def __getitem__(self, index):
        img_index = np.random.randint(0, len(self.img_paths))
        img_path = self.img_paths[img_index]
        img = io.imread(img_path)[:, :, :3]

        img = randomly_rotate_image(img)
        # print(img.shape)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomCrop(self.img_size[0]),
            transforms.ToTensor(),
        ])
        img = transform(img)

        return img, torch.zeros(1, self.img_size[0], self.img_size[1], dtype=torch.long)

    def __len__(self):
        # return len(self.img_paths)
        return self.size


