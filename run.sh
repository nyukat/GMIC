#!/bin/bash

NUM_PROCESSES=10
DEVICE_TYPE='gpu'
NUM_EPOCHS=10
GPU_NUMBER=0

MODEL_PATH='models/sample_model.p'
CROPPED_IMAGE_PATH='sample_data/cropped_images'
EXAM_LIST_PATH='sample_output/data.pkl'
PREDICTIONS_PATH='sample_output/predictions.csv'
PYTHONPATH=$(pwd):$PYTHONPATH

echo 'Run Classifier'
python3 src/modeling/run_model.py \
    --model-path $IMAGE_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $PREDICTIONS_PATH \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER

