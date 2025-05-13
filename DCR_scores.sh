#!/bin/bash

for dataname in adult_equal default_equal shoppers_equal magic_equal beijing_equal news_equal; do
    for model in real test TabKDE simple_KDE smote KDE_VAE_encoding simple_KDE_VAE_encoding diffusion_on_copula tabsyn; do
        echo "Running DCR evaluation for model '$model' on dataset '$dataname'"

        python eval/eval_dcr.py --dataname "$dataname" --model "$model"
        if [ $? -ne 0 ]; then
            echo "Error during eval_dcr.py with dataset '$dataname' and model '$model'"
            exit 1
        fi
    done
done
