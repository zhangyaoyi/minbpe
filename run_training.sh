#!/bin/bash

# Run the training script with nohup in the background
nohup /home/overman/workspaces/stanford-cs336/minbpe/.conda/bin/python /home/overman/workspaces/stanford-cs336/minbpe/train_tokenizer.py > training.log 2>&1 &

# Print the process ID
echo "Training started with PID: $!"
echo "Output is being logged to training.log" 