#!/bin/bash

# Define the lists
models=("coreset" "coreset_random" "simple_KDE_VAE_encoding" "KDE_VAE_encoding" "tabsyn" "TabKDE" "copulaDiff" "smote")
datasets=("adult" "default" "shoppers" "news" "beijing" "magic")

# Store failed runs
failures=()

# Loop over each combination
for model in "${models[@]}"; do
  for data in "${datasets[@]}"; do
    echo "Running model: $model on dataset: $data"
    python eval/eval_quality.py --model "$model" --dataname "$data"
    if [ $? -ne 0 ]; then
      failures+=("$model-$data")
    fi
  done
done

# Report failures
if [ ${#failures[@]} -ne 0 ]; then
  echo -e "\nThe following runs failed:"
  for fail in "${failures[@]}"; do
    echo "$fail"
  done
else
  echo -e "\nAll runs completed successfully."
fi
