import better_exceptions
import random
import math
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from keras.utils import Sequence, to_categorical
import Augmentor





def get_transform_func():
    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.1)
    p.rotate(probability=0.1, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.1, percentage_area=0.95)
    p.random_distortion(probability=0.1, grid_width=2, grid_height=2, magnitude=8)
    p.random_color(probability=0.1, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=0.1, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=0.1, min_factor=0.8, max_factor=1.2)
    p.random_erasing(probability=0.1, rectangle_area=0.2)

    def transform_image(image):
        image = [Image.fromarray(image)]
        for operation in p.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                image = operation.perform_operation(image)
        return image[0]
    return transform_image


class FaceGenerator(Sequence):
    def __init__(self, img_inf, utk_dir=None, batch_size=32, image_size=224):
        self.image_path_and_age = []
        self._load_appa(img_inf)

        self.image_num = len(self.image_path_and_age)
        self.batch_size = batch_size
        self.image_size = image_size
        self.indices = np.random.permutation(self.image_num)
        self.transform_image = get_transform_func()

    def __len__(self):
        return self.image_num // self.batch_size


    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, 1), dtype=np.int32)

        sample_indices = self.indices[idx * batch_size:(idx + 1) * batch_size]

        for i, sample_id in enumerate(sample_indices):
            image_path, age = self.image_path_and_age[sample_id]
            image = cv2.imread(str(image_path))
            x[i] = cv2.resize(image, (image_size, image_size))
            age += math.floor(np.random.randn() * 2 + 0.5)
            y[i] = np.clip(age, 0, 100)

        return x, to_categorical(y, 101)

    def on_epoch_end(self):
        self.indices = np.random.permutation(self.image_num)

    def _load_appa(self, img_inf):
        self.image_path_and_age = img_inf

class ValGenerator(Sequence):
    def __init__(self, img_inf, batch_size=32, image_size=224):
        self.image_path_and_age = []
        self._load_inf(img_inf)
        self.image_num = len(self.image_path_and_age)
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, 1), dtype=np.int32)

        for i in range(batch_size):
            image_path, age = self.image_path_and_age[idx * batch_size + i]
            image = cv2.imread(str(image_path))
            x[i] = cv2.resize(image, (image_size, image_size))
            y[i] = age

        return x, to_categorical(y, 101)

    def _load_inf(self, img_inf):
        self.image_path_and_age = img_inf
