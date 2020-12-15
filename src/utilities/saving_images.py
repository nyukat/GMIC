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

"""
Defines utility functions for saving png and hdf5 images.
"""
import imageio
import h5py


def save_image_as_png(image, filename):
    """
    Saves image as png files while preserving bit depth of the image
    """
    imageio.imwrite(filename, image)


def save_image_as_hdf5(image, filename):
    """
    Saves image as hdf5 files to preserve the floating point values.
    """
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('image', data=image.transpose(), compression="lzf")
    h5f.close()