#!/bin/bash

SEED=42
DEVICE="cuda:0"
LR="1e-5"
BATCH_SIZE=128
EPOCH=10
TEMPERATURE=0.01
SAVE_PER_MODEL="True"
NUMBER_OF_HARD_PROMPT=3
PREFIX_LEN=10
MAX_PROMPT_LEN=20
MAX_GENERATE_LEN=30
OUTPUT_ROOT="./trained_model"

TRAIN_DATASET="mscoco"
SUPPORT_DATASET="mscoco"
TEST_DATASET_1="mscoco"
# TEST_DATASET_2="nocaps"

CURRENT_DATE_TIME=`date "+%Y%m%d-%H%M%S"`
OUTPUT_FILE="${OUTPUT_ROOT}/${TRAIN_DATASET}/${CURRENT_DATE_TIME}"

echo "Begin training ..."
python training.py\
    --seed $SEED\
    --device $DEVICE\
    --lr $LR\
    --batch_size $BATCH_SIZE\
    --epoch $EPOCH\
    --temperature $TEMPERATURE\
    --save_per_model $SAVE_PER_MODEL\
    --retrieve_object_phrase_num_q $NUMBER_OF_HARD_PROMPT\
    --prefix_length $PREFIX_LEN\
    --max_prompt_length $MAX_PROMPT_LEN\
    --max_generate_length $MAX_GENERATE_LEN\
    --output_dir $OUTPUT_FILE\
    --dataset $TRAIN_DATASET
    

echo "Begin inference ${TEST_DATASET_1} ..."
python inference.py\
    --seed $SEED\
    --device $DEVICE\
    --batch_size $BATCH_SIZE\
    --temperature $TEMPERATURE\
    --retrieve_object_phrase_num $NUMBER_OF_HARD_PROMPT\
    --prefix_length $PREFIX_LEN\
    --max_prompt_length $MAX_PROMPT_LEN\
    --max_generate_length $MAX_GENERATE_LEN\
    --test_dataset $TEST_DATASET_1\
    --support_dataset $SUPPORT_DATASET\
    --model_path $OUTPUT_FILE

echo "Done"
