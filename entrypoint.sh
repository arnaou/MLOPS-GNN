#!/bin/sh
echo "I am in the entrypppppoint"
wandb login $YOUR_API_KEY
echo "Pulling data"
dvc pull