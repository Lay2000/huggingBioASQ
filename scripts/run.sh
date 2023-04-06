#!/bin/bash

# Disable tqdm progress bars
# export DISABLE_TQDM=True

# Disable Hugging Face info and warnings
export TRANSFORMERS_VERBOSITY="error"

# Set your desired JSON configuration filename here
JSON_FILENAME="hierarchical.json"

# Run the training script with the specified JSON configuration file
python code/train.py --config configs/${JSON_FILENAME} 2>&1 | tee logs/${JSON_FILENAME%.json}.log