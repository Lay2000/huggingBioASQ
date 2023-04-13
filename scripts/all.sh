#!/bin/bash

# Loop through all JSON files in the "configs" directory
for config_file in configs/xlnet/factoid/*.json
do
  # Extract the filename (without extension) from the path
  filename=$(basename -- "$config_file")
  filename="${filename%.*}"

  # Run the training script with the current JSON file
  python3 code/train.py --config "$config_file" 2>&1 | tee "logs/bert/${filename}.log"
done

for config_file in configs/roberta/*.json
do
  # Extract the filename (without extension) from the path
  filename=$(basename -- "$config_file")
  filename="${filename%.*}"

  # Run the training script with the current JSON file
  python3 code/train.py --config "$config_file" 2>&1 | tee "logs/roberta/${filename}.log"
done

for config_file in configs/xlnet/*.json
do
  # Extract the filename (without extension) from the path
  filename=$(basename -- "$config_file")
  filename="${filename%.*}"

  # Run the training script with the current JSON file
  python3 code/train.py --config "$config_file" 2>&1 | tee "logs/xlnet/${filename}.log"
done
