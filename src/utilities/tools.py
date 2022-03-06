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
Defines utility functions for various tasks in GMIC.
"""
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F



def partition_batch(ls, size):
    """
    Partitions a list into buckets of given maximum length.
    """
    i = 0
    partitioned_lists = []
    while i < len(ls):
        partitioned_lists.append(ls[i: i+size])
        i += size
    return partitioned_lists


def make_sure_in_range(val, min_val, max_val):
    """
    Function that make sure that min < val < max; otherwise return the limit value
    """
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val


def crop(original_img, crop_shape, crop_position, method="center",
         in_place=False, background_val="min"):
    """
    Function that take a crop on the original image.
    This function must staty in numpy since original_img should not be loaded into Pytorch during the network time.
    original_img is large and would consume lots of GPU memory.
    :param original_img:
    :param crop_shape:
    :param crop_position:
    :param method: supported in ["center", "upper_left"]
    :param in_place: if in_place, the effective pixels in the crop will be flagged (1.0) in the original_img
    """
    # retrieve inputs
    I, J = original_img.shape
    crop_x, crop_y = crop_position
    x_delta, y_delta = crop_shape

    # locate the four corners
    if method == "center":
        min_x = int(np.round(crop_x - x_delta / 2))
        max_x = int(np.round(crop_x + x_delta / 2))
        min_y = int(np.round(crop_y - y_delta / 2))
        max_y = int(np.round(crop_y + y_delta/2))
    elif method == "upper_left":
        min_x = int(np.round(crop_x))
        max_x = int(np.round(crop_x + x_delta))
        min_y = int(np.round(crop_y))
        max_y = int(np.round(crop_y + y_delta))

    # make sure that the crops are in range
    min_x = make_sure_in_range(min_x, 0, I)
    max_x = make_sure_in_range(max_x, 0, I)
    min_y = make_sure_in_range(min_y, 0, J)
    max_y = make_sure_in_range(max_y, 0, J)

    # if in_place, flag the original inputs
    if in_place:
        original_img[min_x:max_x, min_y:max_y] = 1.0
    # else make the new matrix
    else:
        # somehow background is normalized to this number
        if background_val == "min":
            output = np.ones(crop_shape) * np.min(original_img)
        else:
            output = np.ones(crop_shape) * background_val
        real_x_delta = max_x - min_x
        real_y_delta = max_y - min_y
        origin_x = crop_shape[0] - real_x_delta
        origin_y = crop_shape[1] - real_y_delta
        output[origin_x:, origin_y:] = original_img[min_x:max_x, min_y:max_y]
        return output


def get_crop_mask(loc, crop_shape, image_shape, method, indicator=True):
    """
    Function that generates the mask
    :param loc:
    :param crop_shape:
    :param image_shape:
    :param method:
    :return:
    """
    crop_map = np.zeros(image_shape)
    for crop_loc in loc:
        # this is the indicator for point of crop
        if indicator:
            crop_map[int(crop_loc[0]), int(crop_loc[1])] = 999.0
        # fill in 1.0 in the cropped regions
        crop(crop_map, crop_shape, crop_loc, method=method, in_place=True)
    return crop_map


def crop_pytorch(original_img_pytorch, crop_shape, crop_position, out,
                 method="center", background_val="min"):
    """
    Function that take a crop on the original image.
    Use PyTorch to do this.
    :param original_img_pytorch: (N,C,H,W) PyTorch Tensor
    :param crop_shape: (h, w) integer tuple
    :param method: supported in ["center", "upper_left"]
    :return: (N, K, h, w) PyTorch Tensor
    """
    # retrieve inputs
    H, W = original_img_pytorch.shape
    crop_x, crop_y = crop_position
    x_delta, y_delta = crop_shape

    # locate the four corners
    if method == "center":
        min_x = int(np.round(crop_x - x_delta / 2))
        max_x = int(np.round(crop_x + x_delta / 2))
        min_y = int(np.round(crop_y - y_delta / 2))
        max_y = int(np.round(crop_y + y_delta / 2))
    elif method == "upper_left":
        min_x = int(np.round(crop_x))
        max_x = int(np.round(crop_x + x_delta))
        min_y = int(np.round(crop_y))
        max_y = int(np.round(crop_y + y_delta))

    # make sure that the crops are in range
    min_x = make_sure_in_range(min_x, 0, H)
    max_x = make_sure_in_range(max_x, 0, H)
    min_y = make_sure_in_range(min_y, 0, W)
    max_y = make_sure_in_range(max_y, 0, W)

    # somehow background is normalized to this number
    if background_val == "min":
        out[:, :] = original_img_pytorch.min()
    else:
        out[:, :] = background_val
    real_x_delta = max_x - min_x
    real_y_delta = max_y - min_y
    origin_x = crop_shape[0] - real_x_delta
    origin_y = crop_shape[1] - real_y_delta
    out[origin_x:, origin_y:] = original_img_pytorch[min_x:max_x, min_y:max_y]


def get_max_window(input_image, window_shape, pooling_logic="avg"):
    """
    Function that makes a sliding window of size window_shape over the
    input_image and return the UPPER_LEFT corner index with max sum
    :param input_image: N*C*H*W
    :param window_shape: h*w
    :return: N*C*2 tensor
    """
    N, C, H, W = input_image.size()
    if pooling_logic == "avg":
        # use average pooling to locate the window sums
        pool_map = torch.nn.functional.avg_pool2d(input_image, window_shape, stride=1)
    elif pooling_logic in ["std", "avg_entropy"]:
        # create sliding windows
        output_size = (H - window_shape[0] + 1, W - window_shape[1] + 1)
        sliding_windows = F.unfold(input_image, kernel_size=window_shape).view(N,C, window_shape[0]*window_shape[1], -1)
        # apply aggregation function on each sliding windows
        if pooling_logic == "std":
            agg_res = sliding_windows.std(dim=2, keepdim=False)
        elif pooling_logic == "avg_entropy":
            agg_res = -sliding_windows*torch.log(sliding_windows)-(1-sliding_windows)*torch.log(1-sliding_windows)
            agg_res = agg_res.mean(dim=2, keepdim=False)
        # merge back
        pool_map = F.fold(agg_res, kernel_size=(1, 1), output_size=output_size)
    _, _, _, W_map = pool_map.size()
    # transform to linear and get the index of the max val locations
    _, max_linear_idx = torch.max(pool_map.view(N, C, -1), -1)
    # convert back to 2d index
    max_idx_x = max_linear_idx // W_map
    max_idx_y = max_linear_idx - max_idx_x * W_map
    # put together the 2d index
    upper_left_points = torch.cat([max_idx_x.unsqueeze(-1), max_idx_y.unsqueeze(-1)], dim=-1)
    return upper_left_points


def generate_mask_uplft(input_image, window_shape, upper_left_points, gpu_number):
    """
    Function that generates mask that sets crops given upper_left
    corners to 0
    :param input_image:
    :param window_shape:
    :param upper_left_points:
    """
    N, C, H, W = input_image.size()
    window_h, window_w = window_shape
    # get the positions of masks
    mask_x_min = upper_left_points[:,:,0]
    mask_x_max = upper_left_points[:,:,0] + window_h
    mask_y_min = upper_left_points[:,:,1]
    mask_y_max = upper_left_points[:,:,1] + window_w
    # generate masks
    mask_x = Variable(torch.arange(0, H).view(-1, 1).repeat(N, C, 1, W))
    mask_y = Variable(torch.arange(0, W).view(1, -1).repeat(N, C, H, 1))
    if gpu_number is not None:
        device = torch.device("cuda:{}".format(gpu_number))
        mask_x = mask_x.cuda().to(device)
        mask_y = mask_y.cuda().to(device)
    x_gt_min = mask_x.float() >= mask_x_min.unsqueeze(-1).unsqueeze(-1).float()
    x_ls_max = mask_x.float() < mask_x_max.unsqueeze(-1).unsqueeze(-1).float()
    y_gt_min = mask_y.float() >= mask_y_min.unsqueeze(-1).unsqueeze(-1).float()
    y_ls_max = mask_y.float() < mask_y_max.unsqueeze(-1).unsqueeze(-1).float()

    # since logic operation is not supported for variable
    # I used * for logic ANd
    selected_x = x_gt_min * x_ls_max
    selected_y = y_gt_min * y_ls_max
    selected = selected_x * selected_y
    mask = 1 - selected.float()
    return mask
