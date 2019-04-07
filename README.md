# Globally-Aware Multiple Instance Classifier forBreast Cancer Screening

## TODO: 
- change license informaiton in each file

## Introduction
This is an implementation of the Globally-Aware Multiple Instance Classifier (GMIC) model as described in our paper [TODO](https://todo). The implementation allows users to get breast cancer predictions and visualization of saliency maps by applying one of our pretrained models. This model is implemented in PyTorch. 


![alt text](https://github.com/nyukat/GMIC/blob/master/sample_data/sample_visualization.png)


* Input: A mammography that is cropped to 2944 x 19202 and are saved as 16-bit png file. As a part of this repository, we provide 4 sample exams (in `sample_data/cropped_images` directory and exam list stored in `sample_data/data.pkl`) each of which includes 2 CC view mammography images and 2 MLO view mammography images.

* Output: The GMIC model generates one prediction for each image: probability of benign and malignant findings. All predictions are saved into a csv file `$OUTPUT_PATH/predictions.csv` that contains the following columns: image_index, benign_pred, malignant_pred, benign_label, malignant_label. In addition, each input image is assoicated with a visualization file saved under `$OUTPUT_PATH/visualization`. An examplar visualization file is illustrated below. The 10 columns (from left to right) represents:
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

## License

This repository is licensed under the terms of the GNU AGPLv3 license.

## How to run the code

`run.sh` will automatically run the entire pipeline and save the prediction results in csv. 

We recommend running the code with a gpu (set by default). To run the code with cpu only, please change `DEVICE_TYPE` in run.sh to 'cpu'.  

If running the individual Python scripts, please include the path to this repository in your `PYTHONPATH` . 

You should obtain the following outputs for the sample exams provided in the repository. 

TODO


## Data

To use one of the pretrained models, the input is TODO. 

The original 12-bit mammograms are saved as rescaled 16-bit images to preserve the granularity of the pixel intensities, while still being correctly displayed in image viewers.

`sample_data/data.pkl` contains a list of exam information. Each exam is represented as a dictionary with the following format:

```python
{
  'horizontal_flip': 'NO',
  'L-CC': ['0_L_CC'],
  'R-CC': ['0_R_CC'],
  'L-MLO': ['0_L_MLO'],
  'R-MLO': ['0_R_MLO'],
  'window_location': {
    'L-CC': [(353, 4009, 0, 2440)],
    'R-CC': [(71, 3771, 952, 3328)],
    'L-MLO': [(0, 3818, 0, 2607)],
    'R-MLO': [(0, 3724, 848, 3328)]
   },
  'rightmost_points': {
    'L-CC': [((1879, 1958), 2389)],
    'R-CC': [((2207, 2287), 2326)],
    'L-MLO': [((2493, 2548), 2556)],
    'R-MLO': [((2492, 2523), 2430)]
   },
  'bottommost_points': {
    'L-CC': [(3605, (100, 100))],
    'R-CC': [(3649, (101, 106))],
    'L-MLO': [(3767, (1456, 1524))],
    'R-MLO': [(3673, (1164, 1184))]
   },
  'distance_from_starting_side': {
    'L-CC': [0],
    'R-CC': [0],
    'L-MLO': [0],
    'R-MLO': [0]
   },
  'best_center': {
    'L-CC': [(1850, 1417)],
    'R-CC': [(2173, 1354)],
    'L-MLO': [(2279, 1681)],
    'R-MLO': [(2185, 1555)]
   }
}
```
We expect images from `L-CC` and `L-MLO` views to be facing right direction, and images from `R-CC` and `R-MLO` views are facing left direction. `horizontal_flip` indicates whether all images in the exam are flipped horizontally from expected. Values for `L-CC`, `R-CC`, `L-MLO`, and `R-MLO` are list of image filenames without extension and directory name. 

Additional information for each image are included as a dictionary. Such dictionary has all 4 views as keys, and the values are the additional information for the corresponding key. 

The additional information includes the following:
- `window_location`: location of cropping window w.r.t. original dicom image so that segmentation map can be cropped in the same way for training.
- `rightmost_points`: rightmost nonzero pixels after correctly being flipped.
- `bottommost_points`: bottommost nonzero pixels after correctly being flipped.
- `distance_from_starting_side`: records if zero-value gap between the edge of the image and the breast is found in the side where the breast starts to appear and thus should have been no gap. Depending on the dataset, this value can be used to determine wrong value of `horizontal_flip`.


The labels for the included exams are as follows:

TODO


## Pipeline

TODO

The following variables defined in `run.sh` can be modified as needed:
* `NUM_PROCESSES`: The number of processes to be used in preprocessing (`src/cropping/crop_mammogram.py` and `src/optimal_centers/get_optimal_centers.py`). Default: 10.
* `DEVICE_TYPE`: Device type to use in heatmap generation and classifiers, either 'cpu' or 'gpu'. Default: 'gpu'
* `NUM_EPOCHS`: The number of epochs to be averaged in the output of the classifiers. Default: 10.
* `GPU_NUMBER`: Specify which one of the GPUs to use when multiple GPUs are available. Default: 0. 
* `MODEL_PATH`: The path where the saved weights model is saved.
* `CROPPED_IMAGE_PATH`: The directory where cropped mammograms are saved.
* `EXAM_LIST_PATH`: The path where the exam list is stored.
* `PREDICTIONS_PATH`: The path to save predictions.
* `VISUALIZATION_PATH`: The path to save visualizations.


### Running the models

`src/modeling/run_model.py`

#### Run the model
```bash
python3 src/modeling/run_model.py \
    --model-path $IMAGE_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $IMAGE_PREDICTIONS_PATH \
    --vis-path $VISUALIZATION_PATH$ \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER
```

This command makes predictions only using images for `$NUM_EPOCHS` epochs with random augmentation and outputs averaged predictions per exam to `$IMAGE_PREDICTIONS_PATH`. 

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
