# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, 
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, 
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, 
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, 
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, 
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of breast_cancer_classifier.
#
# breast_cancer_classifier is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# breast_cancer_classifier is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with breast_cancer_classifier.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================
"""
Runs the model on sample data.
"""
import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tqdm
import cv2
import matplotlib.cm as cm

import src.utilities.pickling as pickling
import src.utilities.tools as tools
import src.modeling.models as models
import src.data_loading.loading as loading
from src.constants import VIEWS


def visualize_example(input_img, saliency_maps, true_segs,
                      patch_locations, patch_img, patch_attentions,
                      save_dir, parameters):
    """
    Function that visualizes the saliency maps for an example
    """
    # colormap lists
    _, _, h, w = saliency_maps.shape
    _, _, H, W = input_img.shape

    # set up colormaps for benign and malignant
    alphas = np.abs(np.linspace(0, 0.95, 259))
    alpha_green = plt.cm.get_cmap('Greens')
    alpha_green._init()
    alpha_green._lut[:, -1] = alphas
    alpha_red = plt.cm.get_cmap('Reds')
    alpha_red._init()
    alpha_red._lut[:, -1] = alphas

    # create visualization template
    total_num_subplots = 4 + parameters["K"]
    figure = plt.figure(figsize=(30, 3))
    # input image + segmentation map
    subfigure = figure.add_subplot(1, total_num_subplots, 1)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    benign_seg, malignant_seg = true_segs
    if benign_seg is not None:
        cm.Greens.set_under('w', alpha=0)
        subfigure.imshow(benign_seg, alpha=0.85, cmap=cm.Greens, clim=[0.9, 1])
    if malignant_seg is not None:
        cm.OrRd.set_under('w', alpha=0)
        subfigure.imshow(malignant_seg, alpha=0.85, cmap=cm.OrRd, clim=[0.9, 1])
    subfigure.set_title("input image")
    subfigure.axis('off')

    # patch map
    subfigure = figure.add_subplot(1, total_num_subplots, 2)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    cm.YlGnBu.set_under('w', alpha=0)
    crop_mask = tools.get_crop_mask(
        patch_locations[0, np.arange(parameters["K"]), :],
        parameters["crop_shape"], (H, W),
        "upper_left")
    subfigure.imshow(crop_mask, alpha=0.7, cmap=cm.YlGnBu, clim=[0.9, 1])
    if benign_seg is not None:
        cm.Greens.set_under('w', alpha=0)
        subfigure.imshow(benign_seg, alpha=0.85, cmap=cm.Greens, clim=[0.9, 1])
    if malignant_seg is not None:
        cm.OrRd.set_under('w', alpha=0)
        subfigure.imshow(malignant_seg, alpha=0.85, cmap=cm.OrRd, clim=[0.9, 1])
    subfigure.set_title("patch map")
    subfigure.axis('off')

    # class activation maps
    subfigure = figure.add_subplot(1, total_num_subplots, 4)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.set_title("SM: malignant")
    subfigure.axis('off')

    subfigure = figure.add_subplot(1, total_num_subplots, 3)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_benign = cv2.resize(saliency_maps[0,0,:,:], (W, H))
    subfigure.imshow(resized_cam_benign, cmap=alpha_green, clim=[0.0, 1.0])
    subfigure.set_title("SM: benign")
    subfigure.axis('off')


    # crops
    for crop_idx in range(parameters["K"]):
        subfigure = figure.add_subplot(1, total_num_subplots, 5 + crop_idx)
        subfigure.imshow(patch_img[0, crop_idx, :, :], cmap='gray', alpha=.8, interpolation='nearest',
                         aspect='equal')
        subfigure.axis('off')
        # crops_attn can be None when we only need the left branch + visualization
        subfigure.set_title("$\\alpha_{0} = ${1:.2f}".format(crop_idx, patch_attentions[crop_idx]))
    plt.savefig(save_dir, bbox_inches='tight', format="png", dpi=500)
    plt.close()


def fetch_cancer_label_by_view(view, cancer_label):
    """
    Function that fetches cancer label using a view
    """
    if view in ["L-CC", "L-MLO"]:
        return cancer_label["left_benign"], cancer_label["left_malignant"]
    elif view in ["R-CC", "R-MLO"]:
        return cancer_label["right_benign"], cancer_label["right_malignant"]


def run_model(model, exam_list, parameters):
    """
    Run the model over images in sample_data.
    Save the predictions as csv and visualizations as png.
    """
    if (parameters["device_type"] == "gpu") and torch.has_cudnn:
        device = torch.device("cuda:{}".format(parameters["gpu_number"]))
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # initialize data holders
    pred_dict = {"image_index": [], "benign_pred": [], "malignant_pred": [],
     "benign_label": [], "malignant_label": []}
    with torch.no_grad():
        # iterate through each exam
        for datum in tqdm.tqdm(exam_list):
            for view in VIEWS.LIST:
                short_file_path = datum[view][0]
                # load image
                # the image is already flipped so no need to do it again
                loaded_image = loading.load_image(
                    image_path=os.path.join(parameters["image_path"], short_file_path + ".png"),
                    view=view,
                    horizontal_flip=False,
                )
                loading.standard_normalize_single_image(loaded_image)
                # load segmentation if available
                benign_seg_path = os.path.join(parameters["segmentation_path"], "{0}_{1}".format(short_file_path, "benign.png"))
                malignant_seg_path = os.path.join(parameters["segmentation_path"], "{0}_{1}".format(short_file_path, "malignant.png"))
                benign_seg = None
                malignant_seg = None
                if os.path.exists(benign_seg_path):
                    loaded_seg = loading.load_image(
                        image_path=benign_seg_path,
                        view=view,
                        horizontal_flip=False,
                    )
                    benign_seg = loaded_seg
                if os.path.exists(malignant_seg_path):
                    loaded_seg = loading.load_image(
                        image_path=malignant_seg_path,
                        view=view,
                        horizontal_flip=False,
                    )
                    malignant_seg = loaded_seg
                # convert python 2D array into 4D torch tensor in N,C,H,W format
                loaded_image = np.expand_dims(np.expand_dims(loaded_image, 0), 0).copy()
                tensor_batch = torch.Tensor(loaded_image).to(device)
                # forward propagation
                output = model(tensor_batch)
                pred_numpy = output.data.cpu().numpy()
                benign_pred, malignant_pred = pred_numpy[0, 0], pred_numpy[0, 1]
                # save visualization
                saliency_maps = model.saliency_map.data.cpu().numpy()
                patch_locations = model.patch_locations
                patch_imgs = model.patches
                patch_attentions = model.patch_attns[0, :].data.cpu().numpy()
                save_dir = os.path.join(parameters["output_path"], "visualization", "{0}.png".format(short_file_path))
                visualize_example(loaded_image, saliency_maps, [benign_seg, malignant_seg],
                      patch_locations, patch_imgs, patch_attentions,
                      save_dir, parameters)
                # propagate holders
                benign_label, malignant_label = fetch_cancer_label_by_view(view, datum["cancer_label"])
                pred_dict["image_index"].append(short_file_path)
                pred_dict["benign_pred"].append(benign_pred)
                pred_dict["malignant_pred"].append(malignant_pred)
                pred_dict["benign_label"].append(benign_label)
                pred_dict["malignant_label"].append(malignant_label)
    return pd.DataFrame(pred_dict)


def load_run_save(model_path, data_path, output_path, parameters):
    """
    Outputs the predictions as csv file
    """
    # construct model
    model = models.GMIC(parameters)
    # load parameters
    if parameters["device_type"] == "gpu":
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    # load metadata
    exam_list = pickling.unpickle_from_file(data_path)
    # create output paths
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "visualization"), exist_ok=True)
    # run the model on the dataset
    output_df = run_model(model, exam_list, parameters)
    # save the predictions and visualization
    output_df.to_csv(os.path.join(output_path, "predictions.csv"), index=False, float_format='%.4f')


def main():
    parser = argparse.ArgumentParser(description='Run image-only model or image+heatmap model')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--segmentation-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    args = parser.parse_args()

    parameters = {
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": args.image_path,
        "segmentation_path": args.segmentation_path,
        "output_path": args.output_path,
        # model related hyper-parameters
        "cam_size": (46, 30),
        "percent_t": 0.01343201825,
        "K": 6,
        "crop_shape": (256, 256),
    }
    load_run_save(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        parameters=parameters,
    )


if __name__ == "__main__":
    main()
