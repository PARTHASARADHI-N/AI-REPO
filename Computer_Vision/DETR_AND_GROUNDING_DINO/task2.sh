#!/bin/bash

if [ "$1" == "1" ] && [ "$2" == "2" ]; then
    echo "Performing zero-shot inference..."
    python p2_zero_shot.py  "$3"  "$5"

elif [ "$1" == "2" ]; then
    if [ "$2" == "1" ]; then
        echo "Training with prompt tuning..."
        python p2_train.py  "$3"  "$4"
    elif [ "$2" == "2" ]; then
        echo "Evaluating prompt-tuned model..."
        python p2_infer.py  "$3"   "$5"
    else
        echo "Invalid step for prompt tuning."
        exit 1
    fi
else
    echo "Invalid mode for task2."
    exit 1
fi
