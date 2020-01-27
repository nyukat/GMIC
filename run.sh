#!/bin/bash

NUM_PROCESSES=10
DEVICE_TYPE='gpu'
GPU_NUMBER=7
MODEL_INDEX='ensemble'

MODEL_PATH='models/'
CROPPED_IMAGE_PATH='sample_data/cropped_images'
SEG_PATH='sample_data/segmentation'
EXAM_LIST_PATH='sample_data/data.pkl'
OUTPUT_PATH='sample_output'
export PYTHONPATH=$(pwd):$PYTHONPATH

echo 'Run Classifier'
python3 src/scripts/run_model.py \
    --model-path $MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --segmentation-path $SEG_PATH \
    --output-path $OUTPUT_PATH \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER \
    --model-index $MODEL_INDEX \
    --visualization-flag

