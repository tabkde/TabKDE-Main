#!/bin/bash

for dataname in adult default shoppers magic beijing news; do
# for dataname in news; do
    for model in copulaDiff tabddpm codi great stasy; do
        echo "🚀 Running model '$model' on dataset '$dataname'"

        python eval/eval_density.py --dataname "$dataname" --model "$model"
        exit_code=$?

        if [ $exit_code -ne 0 ]; then
            echo "⚠️  Warning: Failed on dataset '$dataname' with model '$model' (exit code $exit_code). Continuing..."
            continue
        fi
    done
done

echo "✅ All evaluations attempted. Check warnings above for any failures."
