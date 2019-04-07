# Globally-Aware Multiple Instance Classifier forBreast Cancer Screening

## TODO: 
- change license informaiton in each file

## Introduction
This is an implementation of the Globally-Aware Multiple Instance Classifier (GMIC) model as described in our paper [TODO](https://todo). The implementation allows users to get breast cancer predictions and visualization of saliency maps by applying one of our pretrained models. This model is implemented in PyTorch. 


![alt text](https://github.com/nyukat/GMIC/blob/master/sample_data/sample_visualization.png)


* Input: A mammography image that is cropped to 2944 x 19202 and are saved as 16-bit png file. As a part of this repository, we provide 4 sample exams (in `sample_data/cropped_images` directory and exam list stored in `sample_data/data.pkl`) each of which includes 2 CC view mammography images and 2 MLO view mammography images.

* Output: The GMIC model generates one prediction for each image: probability of benign and malignant findings. All predictions are saved into a csv file `$OUTPUT_PATH/predictions.csv` that contains the following columns: image_index, benign_pred, malignant_pred, benign_label, malignant_label. In addition, each input image is assoicated with a visualization file saved under `$OUTPUT_PATH/visualization`. An examplar visualization file is illustrated above. The 10 columns (from left to right) represents:
  * input mammography with ground truth annotation (green=benign, red=malignant)
  * patch map that illustrates the locations of ROI proposal patches (blue squares)
  * saliency map for benign class
  * saliency map for malignant class
  * 6 ROI proposal patches with the associated attention score on top


## Prerequisites

* Python (3.6)
* PyTorch (0.4.1)
* torchvision (0.2.0)
* NumPy (1.14.3)
* SciPy (1.0.0)
* H5py (2.7.1)
* imageio (2.4.1)
* pandas (0.22.0)
* opencv-python (3.4.2)
* tqdm (4.19.8)


## License

This repository is licensed under the terms of the GNU AGPLv3 license.

## How to run the code

`run.sh` will automatically run the entire pipeline and save the prediction results in csv. If running the individual Python scripts, please include the path to this repository in your `PYTHONPATH`. 

We recommend running the code with a gpu. To run the code with cpu only, please change `DEVICE_TYPE` in run.sh to 'cpu'. 

The following variables defined in `run.sh` can be modified as needed:
* `MODEL_PATH`: The path where the saved weights model is saved.
* `CROPPED_IMAGE_PATH`: The directory where cropped mammograms are saved.
* `SEG_PATH`: The directory where ground truth segmenation are saved.
* `EXAM_LIST_PATH`: The path where the exam list is stored.
* `OUTPUT_PATH`: The path where visualization files and predicitons will be saved.
* `DEVICE_TYPE`: Device type to use in heatmap generation and classifiers, either 'cpu' or 'gpu'.
* `GPU_NUMBER`: Specify which one of the GPUs to use when multiple GPUs are available.


You should obtain the following outputs for the sample exams provided in the repository (found in `sample_output/predictions.csv` by default). 

image_index  |  benign_pred  |  malignant_pred  |  benign_label  |  malignant_label
-------------|---------------|------------------|----------------|-----------------
0_L-CC       |  0.0772       |  0.0051          |  0             |  0
0_R-CC       |  0.6104       |  0.0774          |  1             |  0
0_L-MLO      |  0.1629       |  0.1098          |  0             |  0
0_R-MLO      |  0.6559       |  0.0744          |  1             |  0
1_L-CC       |  0.0083       |  0.0010          |  0             |  0
1_R-CC       |  0.1682       |  0.8866          |  0             |  1
1_L-MLO      |  0.0160       |  0.0082          |  0             |  0
1_R-MLO      |  0.1843       |  0.3429          |  0             |  1
2_L-CC       |  0.0298       |  0.0186          |  0             |  0
2_R-CC       |  0.3353       |  0.0575          |  1             |  0
2_L-MLO      |  0.1075       |  0.0224          |  0             |  0
2_R-MLO      |  0.5951       |  0.0215          |  1             |  0
3_L-CC       |  0.2804       |  0.5027          |  0             |  1
3_R-CC       |  0.0931       |  0.0391          |  0             |  0
3_L-MLO      |  0.2546       |  0.5304          |  0             |  1
3_R-MLO      |  0.0555       |  0.0614          |  0             |  0




## Data

The original 12-bit mammograms are saved as rescaled 16-bit images to preserve the granularity of the pixel intensities, while still being correctly displayed in image viewers.

`sample_data/cropped_images` contains 4 exams each of which includes 4 mammography images (L-CC, L-MLO, R-CC, R-MLO). All mammography images are saved in png format.

`sample_data/segmentatio` contains the 


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
In their original formats, images from `L-CC` and `L-MLO` views face right direction, and images from `R-CC` and `R-MLO` views faces left direction. We horizontally flipped `R-CC` and `R-MLO` images so that all four views face right direction. Values for `L-CC`, `R-CC`, `L-MLO`, and `R-MLO` are list of image filenames without extension and directory name. 



## Reference

If you found this code useful, please cite our paper:

**TODO**\
TODO\
2019

    @article{TODO, 
        title = {TODO},
        author = {TODO}, 
        journal = {TODO},
        year = {2019}
    }
