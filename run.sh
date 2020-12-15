#!/bin/bash

NUM_PROCESSES=10
DEVICE_TYPE='cpu'
GPU_NUMBER=0
MODEL_INDEX='1'

MODEL_PATH='models/'
DATA_FOLDER='sample_data/images'
INITIAL_EXAM_LIST_PATH='sample_data/exam_list_before_cropping.pkl'
CROPPED_IMAGE_PATH='sample_output/cropped_images'
CROPPED_EXAM_LIST_PATH='sample_output/cropped_images/cropped_exam_list.pkl'
SEG_PATH='sample_data/segmentation'
EXAM_LIST_PATH='sample_output/data.pkl'
OUTPUT_PATH='sample_output'
export PYTHONPATH=$(pwd):$PYTHONPATH


echo 'Stage 1: Crop Mammograms'
python3 src/cropping/crop_mammogram.py \
    --input-data-folder $DATA_FOLDER \
    --output-data-folder $CROPPED_IMAGE_PATH \
    --exam-list-path $INITIAL_EXAM_LIST_PATH  \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH  \
    --num-processes $NUM_PROCESSES

echo 'Stage 2: Extract Centers'
python3 src/optimal_centers/get_optimal_centers.py \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH \
    --data-prefix $CROPPED_IMAGE_PATH \
    --output-exam-list-path $EXAM_LIST_PATH \
    --num-processes $NUM_PROCESSES

echo 'Stage 3: Run Classifier'
python3 src/scripts/run_model.py \
    --model-path $MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --segmentation-path $SEG_PATH \
    --output-path $OUTPUT_PATH \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER \
    --model-index $MODEL_INDEX \
    #--visualization-flag

