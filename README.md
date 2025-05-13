# TabKDE and Hybrid Models

This repository extends the [TabSyn](https://github.com/amazon-science/tabsyn) codebase by adding:

- **TabKDE** methods
- **Hybrid models** that combine TabKDE and VAE/Diffusion approaches

---

## 📦 Dataset Preparation

1. **Download the Dataset**

   Run:
   ```bash
   python download_dataset.py 
   ```
*(For the IBM dataset, use the link — [credit_card_transactions-ibm_v2.csv](https://www.kaggle.com/code/yichenzhang1226/ibm-credit-card-fraud-detection-eda-random-forest/input?select=credit_card_transactions-ibm_v2.csv).) to download and follow the instruction in the paper to create the tran/test split*


3. **Process the Dataset**

   Run:
   ```bash
   python process_dataset.py --dataname "$dataname"
   ```

---

## 🧱 TabKDE / Simple KDE

**TabKDE:**
```bash
python main.py --dataname "$dataname"  --method TabKDE --mode train
python main.py --dataname "$dataname"  --method TabKDE --mode sample
```

**Simple_KDE (sampling only):**
```bash
python main.py --dataname "$dataname" --method simple_KDE --mode sample
```

---

## 🎯 Coreset Sampling

Requires **TabKDE** training first (uses copula embedding).

```bash
python coreset.py --dataname "$dataname"
python coreset_random.py --dataname "$dataname"
```

---

## 🌐 CopulaDiff

1. **Train Copula Embedding**
```bash
python main.py --dataname "$dataname" --method copula_diff_encoding --mode train
```

2. **Train TabSyn with Copula Encoding**
```bash
python main.py --dataname "$dataname" --method tabsyn --mode train --latent_encoding copula_diff_encoding --num_epochs "$n_epochs"
```

3. **Sample**
```bash
python main.py --dataname "$dataname" --method tabsyn --mode sample --latent_encoding copula_diff_encoding
```

---

## 🔄 TabSyn (Vanilla)

1. **Train VAE**
```bash
python main.py --dataname "$dataname" --method vae --mode train --num_epochs "$n_epochs"
```

2. **Train TabSyn**
```bash
python main.py --dataname "$dataname" --method tabsyn --mode train --num_epochs "$n_epochs"
```

3. **Sample**
```bash
python main.py --dataname "$dataname" --method tabsyn --mode sample
```

---

## 📈 VAE-TabKDE

Requires VAE training (via TabSyn).

```bash
python main.py --dataname "$dataname" --method KDE_VAE_encoding --mode sample
```

---

## 📉 VAE-SimpleKDE

```bash
python main.py --dataname "$dataname" --method simple_KDE_VAE_encoding --mode sample
```

---

## 🧪 Baselines (excluding TabSyn)

```bash
python main.py --dataname "$dataname" --method "$baseline" --mode train
python main.py --dataname "$dataname" --method "$baseline" --mode sample
```


## Evaluation
We evaluate the quality of synthetic data using metrics from various aspects.

#### Density estimation of single column and pair-wise correlation ([link](https://docs.sdv.dev/sdmetrics/reports/quality-report/whats-included))

```
python eval/eval_density.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA]
```


#### Alpha Precision and Beta Recall ([paper link](https://arxiv.org/abs/2102.08921))
- $\alpha$-preicison: the fidelity of synthetic data
- $\beta$-recall: the diversity of synthetic data

```
python eval/eval_quality.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA]
```

#### Machine Learning Efficiency

```
python eval/eval_mle.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA]
```

#### Pricavy protection: Distance to Closest Record (DCR)

```
python eval/eval_dcr.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA]
```

Note: the optimal DCR score depends on the ratio between #Train and #Holdout (# Test). Ideally, DCR sore should be #Train / (#Train + #Holdout). To let the optimal score be $50\%$, you have to let the training and testing set have the same size. 

#### Detection: Classifier Two Sample Tests (C2ST)

```
python eval/eval_detection.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA]
```


