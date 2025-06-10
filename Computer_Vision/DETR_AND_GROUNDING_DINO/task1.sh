#!/bin/bash

if [ "$1" == "1" ]; then
    echo "Evaluating pretrained model..."
    python p1_infer.py  "$4"  "$5"  "$6"

elif [ "$1" == "2" ]; then
    STRATEGY=$2
    STEP=$3
    if [ "$STEP" == "1" ]; then
        echo "Fine-tuning strategy $STRATEGY - Training..."
        python part1_train.py  "$2"  "$4"  "$5"
    elif [ "$STEP" == "2" ]; then
        echo "Fine-tuning strategy $STRATEGY - Evaluation..."
        python p1_infer.py   "$4"  "$5"  "$6"
    else
        echo "Invalid step for fine-tuning."
        exit 1
    fi
else
    echo "Invalid mode for task1."
    exit 1
fi
