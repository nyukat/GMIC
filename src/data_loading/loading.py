# Copyright (C) 2020 Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu,
# Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of GMIC.
#
# GMIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# GMIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GMIC.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import numpy as np
from src.constants import VIEWS
import imageio
import src.data_loading.augmentations as augmentations


def flip_image(image, view, horizontal_flip):
    """
    If training mode, makes all images face right direction.
    In medical, keeps the original directions unless horizontal_flip is set.
    """
    if horizontal_flip == 'NO':
        if VIEWS.is_right(view):
            image = np.fliplr(image)
    elif horizontal_flip == 'YES':
        if VIEWS.is_left(view):
            image = np.fliplr(image)

    return image


def standard_normalize_single_image(image):
    """
    Standardizes an image in-place 
    """
    image -= np.mean(image)
    image /= np.maximum(np.std(image), 10**(-5))


def read_image_png(file_name):
    image = np.array(imageio.imread(file_name))
    return image


def load_image(image_path, view, horizontal_flip):
    """
    Loads a png or hdf5 image as floats and flips according to its view.
    """
    if image_path.endswith("png"):
        image = read_image_png(image_path)
    else:
        raise RuntimeError()
    image = image.astype(np.float32)
    image = flip_image(image, view, horizontal_flip)
    return image


def process_image(image, view, best_center):
    """
    Applies augmentation window with random noise in location and size
    and return normalized cropped image.
    """
    cropped_image, _ = augmentations.random_augmentation_best_center(
        image=image,
        input_size=(2944, 1920),
        random_number_generator=np.random.RandomState(0),
        best_center=best_center,
        view=view
    )

    # For test time only, normalize a copy of the cropped image
    # in order to avoid changing the value of original image which gets augmented multiple times
    cropped_image = cropped_image.copy()
    standard_normalize_single_image(cropped_image)

    return cropped_image





