#!/bin/bash

for dataname in ibm_func; do
    for model in real test TabKDE smote  diffusion_on_copula; do
        echo "Running DCR evaluation for model '$model' on dataset '$dataname'"

        python eval/eval_dcr_ibm.py --dataname "$dataname" --model "$model"
        if [ $? -ne 0 ]; then
            echo "Error during eval_dcr.py with dataset '$dataname' and model '$model'"
            exit 1
        fi
    done
done
