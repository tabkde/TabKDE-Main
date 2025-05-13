#!/bin/bash

# Define the array of bandwidths
BW=(0.2 0.2 0.15 1.0 0.2 0.2)

# Define the list of dataset names
datasets=("default_equal" "shoppers_equal" "magic_equal" "beijing_equal" "news_equal")

# Loop through the datasets using index
for i in "${!datasets[@]}"; do
    dataname="${datasets[$i]}"
    bw="${BW[$i]}"

    echo "Running Coreset model evaluation on dataset '$dataname' with bandwidth $bw"
    
    python coreset.py --data_name "$dataname" --bandwidth "$bw"
    if [ $? -ne 0 ]; then
        echo "❌ Error during coreset.py execution with dataset '$dataname'"
        exit 1
    fi
done

echo "✅ All evaluations completed successfully."
