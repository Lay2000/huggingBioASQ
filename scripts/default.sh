#!/bin/bash
# Set your desired JSON configuration filename here
JSON_FILENAME="default.json"

# Run the training script with the specified JSON configuration file
python code/train.py --config configs/${JSON_FILENAME} 2>&1 | tee logs/${JSON_FILENAME%.json}.log