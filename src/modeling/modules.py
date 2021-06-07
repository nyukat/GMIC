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
Defines modules for breast cancer classification models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utilities import tools
from torchvision.models.resnet import conv3x3


class BasicBlockV2(nn.Module):
    """
    Basic Residual Block of ResNet V2
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV2, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # Phase 1
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)

        # Phase 2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class BasicBlockV1(nn.Module):
    """
    Basic Residual Block of ResNet V1
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetV2(nn.Module):
    """
    Adapted fom torchvision ResNet, converted to v2
    """
    def __init__(self,
                 input_channels, num_filters,
                 first_layer_kernel_size, first_layer_conv_stride,
                 blocks_per_layer_list, block_strides_list, block_fn,
                 first_layer_padding=0,
                 first_pool_size=None, first_pool_stride=None, first_pool_padding=0,
                 growth_factor=2):
        super(ResNetV2, self).__init__()
        self.first_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=num_filters,
            kernel_size=first_layer_kernel_size,
            stride=first_layer_conv_stride,
            padding=first_layer_padding,
            bias=False,
        )
        self.first_pool = nn.MaxPool2d(
            kernel_size=first_pool_size,
            stride=first_pool_stride,
            padding=first_pool_padding,
        )

        self.layer_list = nn.ModuleList()
        current_num_filters = num_filters
        self.inplanes = num_filters
        for i, (num_blocks, stride) in enumerate(zip(
                blocks_per_layer_list, block_strides_list)):
            self.layer_list.append(self._make_layer(
                block=block_fn,
                planes=current_num_filters,
                blocks=num_blocks,
                stride=stride,
            ))
            current_num_filters *= growth_factor
        self.final_bn = nn.BatchNorm2d(
            current_num_filters // growth_factor * block_fn.expansion
        )
        self.relu = nn.ReLU()

        # Expose attributes for downstream dimension computation
        self.num_filters = num_filters
        self.growth_factor = growth_factor

    def forward(self, x):
        h = self.first_conv(x)
        h = self.first_pool(h)
        for i, layer in enumerate(self.layer_list):
            h = layer(h)
        h = self.final_bn(h)
        h = self.relu(h)
        return h

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
        )

        layers_ = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers_.append(block(self.inplanes, planes))

        return nn.Sequential(*layers_)


class ResNetV1(nn.Module):
    """
    Class that represents a ResNet with classifier sequence removed
    """
    def __init__(self, initial_filters, block, layers, input_channels=1):

        self.inplanes = initial_filters
        self.num_layers = len(layers)
        super(ResNetV1, self).__init__()

        # initial sequence
        # the first sequence only has 1 input channel which is different from original ResNet
        self.conv1 = nn.Conv2d(input_channels, initial_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # residual sequence
        for i in range(self.num_layers):
            num_filters = initial_filters * pow(2,i)
            num_stride = (1 if i == 0 else 2)
            setattr(self, 'layer{0}'.format(i+1), self._make_layer(block, num_filters, layers[i], stride=num_stride))
        self.num_filter_last_seq = initial_filters * pow(2, self.num_layers-1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # first sequence
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # residual sequences
        for i in range(self.num_layers):
            x = getattr(self, 'layer{0}'.format(i+1))(x)
        return x


class DownsampleNetworkResNet18V1(ResNetV1):
    """
    Downsampling using ResNet V1
    First conv is 7*7, stride 2, padding 3, cut 1/2 resolution
    """
    def __init__(self):
        super(DownsampleNetworkResNet18V1, self).__init__(
            initial_filters=64,
            block=BasicBlockV1,
            layers=[2, 2, 2, 2],
            input_channels=3)

    def forward(self, x):
        last_feature_map = super(DownsampleNetworkResNet18V1, self).forward(x)
        return last_feature_map


class AbstractMILUnit:
    """
    An abstract class that represents an MIL unit module
    """
    def __init__(self, parameters, parent_module):
        self.parameters = parameters
        self.parent_module = parent_module


class PostProcessingStandard(nn.Module):
    """
    Unit in Global Network that takes in x_out and produce saliency maps
    """
    def __init__(self, parameters):
        super(PostProcessingStandard, self).__init__()
        # map all filters to output classes
        self.gn_conv_last = nn.Conv2d(parameters["post_processing_dim"],
                                      parameters["num_classes"],
                                      (1, 1), bias=False)

    def forward(self, x_out):
        out = self.gn_conv_last(x_out)
        return torch.sigmoid(out)


class GlobalNetwork(AbstractMILUnit):
    """
    Implementation of Global Network using ResNet-22
    """
    def __init__(self, parameters, parent_module):
        super(GlobalNetwork, self).__init__(parameters, parent_module)
        # downsampling-branch
        if "use_v1_global" in parameters and parameters["use_v1_global"]:
            self.downsampling_branch = DownsampleNetworkResNet18V1()
        else:
            self.downsampling_branch = ResNetV2(input_channels=1, num_filters=16,
                     # first conv layer
                     first_layer_kernel_size=(7,7), first_layer_conv_stride=2,
                     first_layer_padding=3,
                     # first pooling layer
                     first_pool_size=3, first_pool_stride=2, first_pool_padding=0,
                     # res blocks architecture
                     blocks_per_layer_list=[2, 2, 2, 2, 2],
                     block_strides_list=[1, 2, 2, 2, 2],
                     block_fn=BasicBlockV2,
                     growth_factor=2)
        # post-processing
        self.postprocess_module = PostProcessingStandard(parameters)

    def add_layers(self):
        self.parent_module.ds_net = self.downsampling_branch
        self.parent_module.left_postprocess_net = self.postprocess_module

    def forward(self, x):
        # retrieve results from downsampling network at all 4 levels
        last_feature_map = self.downsampling_branch.forward(x)
        # feed into postprocessing network
        cam = self.postprocess_module.forward(last_feature_map)
        return last_feature_map, cam





class TopTPercentAggregationFunction(AbstractMILUnit):
    """
    An aggregator that uses the SM to compute the y_global.
    Use the sum of topK value
    """
    def __init__(self, parameters, parent_module):
        super(TopTPercentAggregationFunction, self).__init__(parameters, parent_module)
        self.percent_t = parameters["percent_t"]
        self.parent_module = parent_module

    def forward(self, cam):
        batch_size, num_class, H, W = cam.size()
        cam_flatten = cam.view(batch_size, num_class, -1)
        top_t = int(round(W*H*self.percent_t))
        selected_area = cam_flatten.topk(top_t, dim=2)[0]
        return selected_area.mean(dim=2)


class RetrieveROIModule(AbstractMILUnit):
    """
    A Regional Proposal Network instance that computes the locations of the crops
    Greedy select crops with largest sums
    """
    def __init__(self, parameters, parent_module):
        super(RetrieveROIModule, self).__init__(parameters, parent_module)
        self.crop_method = "upper_left"
        self.num_crops_per_class = parameters["K"]
        self.crop_shape = parameters["crop_shape"]
        self.gpu_number = None if parameters["device_type"]!="gpu" else parameters["gpu_number"]

    def forward(self, x_original, cam_size, h_small):
        """
        Function that use the low-res image to determine the position of the high-res crops
        :param x_original: N, C, H, W pytorch tensor
        :param cam_size: (h, w)
        :param h_small: N, C, h_h, w_h pytorch tensor
        :return: N, num_classes*k, 2 numpy matrix; returned coordinates are corresponding to x_small
        """
        # retrieve parameters
        _, _, H, W = x_original.size()
        (h, w) = cam_size
        N, C, h_h, w_h = h_small.size()

        # make sure that the size of h_small == size of cam_size
        assert h_h == h, "h_h!=h"
        assert w_h == w, "w_h!=w"

        # adjust crop_shape since crop shape is based on the original image
        crop_x_adjusted = int(np.round(self.crop_shape[0] * h / H))
        crop_y_adjusted = int(np.round(self.crop_shape[1] * w / W))
        crop_shape_adjusted = (crop_x_adjusted, crop_y_adjusted)

        # greedily find the box with max sum of weights
        current_images = h_small
        all_max_position = []
        # combine channels
        max_vals = current_images.view(N, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        min_vals = current_images.view(N, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        range_vals = max_vals - min_vals
        normalize_images = current_images - min_vals
        normalize_images = normalize_images / range_vals
        current_images = normalize_images.sum(dim=1, keepdim=True)

        for _ in range(self.num_crops_per_class):
            max_pos = tools.get_max_window(current_images, crop_shape_adjusted, "avg")
            all_max_position.append(max_pos)
            mask = tools.generate_mask_uplft(current_images, crop_shape_adjusted, max_pos, self.gpu_number)
            current_images = current_images * mask
        return torch.cat(all_max_position, dim=1).data.cpu().numpy()


class LocalNetwork(AbstractMILUnit):
    """
    The local network that takes a crop and computes its hidden representation
    Use ResNet
    """
    def add_layers(self):
        """
        Function that add layers to the parent module that implements nn.Module
        :return:
        """
        self.parent_module.dn_resnet = ResNetV1(64, BasicBlockV1, [2,2,2,2], 3)

    def forward(self, x_crop):
        """
        Function that takes in a single crop and return the hidden representation
        :param x_crop: (N,C,h,w)
        :return:
        """
        # forward propagte using ResNet
        res = self.parent_module.dn_resnet(x_crop.expand(-1, 3, -1 , -1))
        # global average pooling
        res = res.mean(dim=2).mean(dim=2)
        return res


class AttentionModule(AbstractMILUnit):
    """
    The attention module takes multiple hidden representations and compute the attention-weighted average
    Use Gated Attention Mechanism in https://arxiv.org/pdf/1802.04712.pdf
    """
    def add_layers(self):
        """
        Function that add layers to the parent module that implements nn.Module
        :return:
        """
        # The gated attention mechanism
        self.parent_module.mil_attn_V = nn.Linear(512, 128, bias=False)
        self.parent_module.mil_attn_U = nn.Linear(512, 128, bias=False)
        self.parent_module.mil_attn_w = nn.Linear(128, 1, bias=False)
        # classifier
        self.parent_module.classifier_linear = nn.Linear(512, self.parameters["num_classes"], bias=False)

    def forward(self, h_crops):
        """
        Function that takes in the hidden representations of crops and use attention to generate a single hidden vector
        :param h_small:
        :param h_crops:
        :return:
        """
        batch_size, num_crops, h_dim = h_crops.size()
        h_crops_reshape = h_crops.view(batch_size * num_crops, h_dim)
        # calculate the attn score
        attn_projection = torch.sigmoid(self.parent_module.mil_attn_U(h_crops_reshape)) * \
                          torch.tanh(self.parent_module.mil_attn_V(h_crops_reshape))
        attn_score = self.parent_module.mil_attn_w(attn_projection)
        # use softmax to map score to attention
        attn_score_reshape = attn_score.view(batch_size, num_crops)
        attn = F.softmax(attn_score_reshape, dim=1)

        # final hidden vector
        z_weighted_avg = torch.sum(attn.unsqueeze(-1) * h_crops, 1)

        # map to the final layer
        y_crops = torch.sigmoid(self.parent_module.classifier_linear(z_weighted_avg))
        return z_weighted_avg, attn, y_crops



