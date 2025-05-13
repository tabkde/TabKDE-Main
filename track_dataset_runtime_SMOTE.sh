#!/bin/bash
set -e

# Track which step is currently running
current_step="initialising"
dataname="(not yet started)"  # Set default so error message is meaningful
trap 'echo "❌ Script failed while processing dataset: $dataname | Step: $current_step"; exit 1' ERR

csvfile="per_dataset_runtime_log_smote.csv"
echo "Dataset,SMOTE Sampling" > "$csvfile"

for dataname in adult_equal default_equal shoppers_equal beijing_equal news_equal magic_equal; do
    echo "🚀 Running SMOTE sampling for: $dataname"

    t1=$(date +%s)

    current_step="SMOTE Sampling"
    python main.py --dataname "$dataname" --method smote --mode sample

    t2=$(date +%s)
    smote_duration=$((t2 - t1))

    echo "$dataname,$smote_duration" >> "$csvfile"

    # Reset step for next dataset
    current_step="done"
done

echo "✅ All timing details saved to $csvfile"
