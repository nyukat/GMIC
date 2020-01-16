#!/bin/bash

NUM_PROCESSES=10
DEVICE_TYPE='gpu'
GPU_NUMBER=0

#MODEL_PATH='models/sample_model.p'
MODEL_PATH='models/new_model.p'
CROPPED_IMAGE_PATH='sample_data/cropped_images'
SEG_PATH='sample_data/segmentation'
EXAM_LIST_PATH='sample_data/data.pkl'
OUTPUT_PATH='sample_output'
#PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

echo 'Run Classifier'
python3 src/scripts/run_model.py \
    --model-path $MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --segmentation-path $SEG_PATH \
    --output-path $OUTPUT_PATH \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER

