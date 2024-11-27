#!/bin/bash

# Default parameter values
BATCH_SIZE=8
NUM_EPOCH=60
IMAGE_HEIGHT=256
IMAGE_WIDTH=256
TEST_SET_RATIO=0.2
RANDOM_SEED=99
LEARNING_RATE=0.0001
RHO=3.0
MODEL="MySimpleModel"  #  MySimpleModel, InceptionV3,ResNet50,DenseNet121,VGG16,EfficientNetB0
# Run the Python script with the specified parameters
python main.py \
    --batch_size $BATCH_SIZE \
    --num_epoch $NUM_EPOCH \
    --image_height $IMAGE_HEIGHT \
    --image_weight $IMAGE_WIDTH \
    --test_set_ratio $TEST_SET_RATIO \
    --random_seed $RANDOM_SEED \
    --learning_rate $LEARNING_RATE \
    --rho $RHO \
    --model $MODEL