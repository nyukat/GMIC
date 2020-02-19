# An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization

## Introduction
This is an implementation of the Globally-Aware Multiple Instance Classifier (GMIC) model as described in [our paper](https://arxiv.org/abs/2002.07613). The architecture of the proposed model is shown below.

<p align="center">
  <img width="793" height="729" src="https://github.com/nyukat/GMIC/blob/master/mia_structure.png">
</p>

Highlights of GMIC:
- **High Accuracy**: GMIC outperformed ResNet-34 and Faster R-CNN.
- **High Efficiency**: Compared to ResNet-34, GMIC has **28.8%** fewer parameters, uses **78.43%** less GPU memory and is **4.1x** faster during inference and **5.6x** faster during training.
- **Weakly Supervised Lesion Localization**: Despite being trained with only image-level labels indicating the presence of any benign or malignant lesion, GMIC is able to generate pixel-level saliency maps (shown below) that provide additional interpretability.

The implementation allows users to obtain breast cancer predictions and visualization of saliency maps by applying one of our pretrained models. We provide weights for 5 GMIC-ResNet-18 models. The model is implemented in PyTorch. 

* Input: A mammography image that is cropped to 2944 x 1920 and are saved as 16-bit png files. As a part of this repository, we provide 4 sample exams (in `sample_data/cropped_images` directory and exam list stored in `sample_data/data.pkl`), each of which includes 2 CC view images and 2 MLO view images.

* Output: The GMIC model generates one prediction for each image: probability of benign and malignant findings. All predictions are saved into a csv file `$OUTPUT_PATH/predictions.csv` that contains the following columns: image_index, benign_pred, malignant_pred, benign_label, malignant_label. In addition, each input image is associated with a visualization file saved under `$OUTPUT_PATH/visualization`. An exemplar visualization file is illustrated below. The images (from left to right) represent:
  * input mammography with ground truth annotation (green=benign, red=malignant),
  * patch map that illustrates the locations of ROI proposal patches (blue squares),
  * saliency map for benign class,
  * saliency map for malignant class,
  * 6 ROI proposal patches with the associated attention score on top.
  
![alt text](https://github.com/nyukat/GMIC/blob/master/sample_data/sample_visualization.png)



## Prerequisites

* Python (3.6)
* PyTorch (1.1.0)
* torchvision (0.2.0)
* NumPy (1.14.3)
* SciPy (1.0.0)
* H5py (2.7.1)
* imageio (2.4.1)
* pandas (0.22.0)
* opencv-python (3.4.2)
* tqdm (4.19.8)
* matplotlib (3.0.2)


## License

This repository is licensed under the terms of the GNU AGPLv3 license.

## How to run the code

You need to first install conda in your environment. **Before running the code, please run `pip install -r requirements.txt` first.** Once you have installed all the dependencies, `run.sh` will automatically run the entire pipeline and save the prediction results in csv. Note that you need to first cd to the project directory and then execute `. ./run.sh`. When running the individual Python scripts, please include the path to this repository in your `PYTHONPATH`. 

We recommend running the code with a GPU. To run the code with CPU only, please change `DEVICE_TYPE` in run.sh to 'cpu'. 

The following variables defined in `run.sh` can be modified as needed:
* `MODEL_PATH`: The path where the model weights are saved.
* `CROPPED_IMAGE_PATH`: The directory where cropped mammograms are saved.
* `SEG_PATH`: The directory where ground truth segmenations are saved.
* `EXAM_LIST_PATH`: The path where the exam list is stored.
* `OUTPUT_PATH`: The path where visualization files and predictions will be saved.
* `DEVICE_TYPE`: Device type to use in heatmap generation and classifiers, either 'cpu' or 'gpu'.
* `GPU_NUMBER`: GPUs number multiple GPUs are available.
* `MODEL_INDEX`: Which one of the five models to use. Valid values include {'1', '2', '3', '4', '5','ensemble'}.
* `visualization-flag`: Whether to generate visualization.


You should obtain the following outputs for the sample exams provided in the repository (found in `sample_output/predictions.csv` by default). 

image_index  |  benign_pred  |  malignant_pred  |  benign_label  |  malignant_label
-------------|---------------|------------------|----------------|-----------------
0_L-CC       |  0.1356       |  0.0081          |  0             |  0
0_R-CC       |  0.8929       |  0.3259          |  1             |  0
0_L-MLO      |  0.2368       |  0.0335          |  0             |  0
0_R-MLO      |  0.9509       |  0.1812          |  1             |  0
1_L-CC       |  0.0546       |  0.0168          |  0             |  0
1_R-CC       |  0.5986       |  0.9910          |  0             |  1
1_L-MLO      |  0.0414       |  0.0139          |  0             |  0
1_R-MLO      |  0.5383       |  0.9308          |  0             |  1
2_L-CC       |  0.0678       |  0.0227          |  0             |  0
2_R-CC       |  0.1917       |  0.0603          |  1             |  0
2_L-MLO      |  0.1210       |  0.0093          |  0             |  0
2_R-MLO      |  0.2440       |  0.0231          |  1             |  0
3_L-CC       |  0.6295       |  0.9326          |  0             |  1
3_R-CC       |  0.2291       |  0.1603          |  0             |  0
3_L-MLO      |  0.6304       |  0.7496          |  0             |  1
3_R-MLO      |  0.0622       |  0.0507          |  0             |  0


## Data

`sample_data/cropped_images` contains 4 exams each of which includes 4 mammography images (L-CC, L-MLO, R-CC, R-MLO). All mammography images are saved in png format. The original 12-bit mammograms are saved as rescaled 16-bit images to preserve the granularity of the pixel intensities, while still being correctly displayed in image viewers.

`sample_data/segmentation` contains the binary pixel-level segmentation labels for some exams. All segmentations are saved as png images.

`sample_data/data.pkl` contains a list of exam information. Each exam is represented as a dictionary with the following format:

```python
{'horizontal_flip': 'NO',
  'L-CC': ['0_L-CC'],
  'L-MLO': ['0_L-MLO'],
  'R-MLO': ['0_R-MLO'],
  'R-CC': ['0_R-CC'],
  'best_center': {'R-CC': [(1136.0, 158.0)],
   'R-MLO': [(1539.0, 252.0)],
   'L-MLO': [(1530.0, 307.0)],
   'L-CC': [(1156.0, 262.0)]},
  'cancer_label': {'benign': 1,
   'right_benign': 0,
   'malignant': 0,
   'left_benign': 1,
   'unknown': 0,
   'right_malignant': 0,
   'left_malignant': 0},
  'L-CC_benign_seg': ['0_L-CC_benign'],
  'L-CC_malignant_seg': ['0_L-CC_malignant'],
  'L-MLO_benign_seg': ['0_L-MLO_benign'],
  'L-MLO_malignant_seg': ['0_L-MLO_malignant'],
  'R-MLO_benign_seg': ['0_R-MLO_benign'],
  'R-MLO_malignant_seg': ['0_R-MLO_malignant'],
  'R-CC_benign_seg': ['0_R-CC_benign'],
  'R-CC_malignant_seg': ['0_R-CC_malignant']}
```
In their original formats, images from `L-CC` and `L-MLO` views face right, and images from `R-CC` and `R-MLO` views face left. We horizontally flipped `R-CC` and `R-MLO` images so that all four views face right. Values for `L-CC`, `R-CC`, `L-MLO`, and `R-MLO` are list of image filenames without extensions and directory name. 



## Reference

If you found this code useful, please cite our paper:

**An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization**\
Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu, Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho and Krzysztof J. Geras\
arXiv:2002.07613, 2020.
    
    @article{shen2020an, 
    title={An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization},
        author={Shen, Yiqiu and Wu, Nan and Phang, Jason and Park, Jungkyu and Liu, Kangning and Tyagi, Sudarshini and Heacock, Laura and Kim, Gene and Moy, Linda and Cho, Kyunghyun and Geras, Krzysztof J},
        journal={arXiv:2002.07613},
        year={2020}}


Reference to previous GMIC version:

**Globally-Aware Multiple Instance Classifier for Breast Cancer Screening**\
Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, S. Gene Kim, Linda Moy, Kyunghyun Cho and Krzysztof J. Geras\
Machine Learning in Medical Imaging - 10th International Workshop, MLMI 2019, Held in Conjunction with MICCAI 2019, Proceedings. Springer , 2019. p. 18-26 (Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics); Vol. 11861 LNCS).
    
    @inproceedings{shen2019globally, 
    title={Globally-Aware Multiple Instance Classifier for Breast Cancer Screening},
        author={Shen, Yiqiu and Wu, Nan and Phang, Jason and Park, Jungkyu and Kim, Gene and Moy, Linda and Cho, Kyunghyun and Geras, Krzysztof J},
        booktitle={Machine Learning in Medical Imaging: 10th International Workshop, MLMI 2019, Held in Conjunction with MICCAI 2019, Shenzhen, China, October 13, 2019, Proceedings},
        volume={11861},
        pages={18-26},
        year={2019},
        organization={Springer Nature}}
