#!/bin/bash

if [ "$1" == "1" ]; then
    echo "Training for task 3..."
    python p3_finetune.py  "$2" "$3"

elif [ "$1" == "2" ]; then
    echo "Evaluating for task 3..."
    python p3_infer.py  "$2" "$3" "$4"

else
    echo "Invalid mode for task3."
    exit 1
fi
