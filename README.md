# TabKDE and Hybrid Models

This repository extends the [TabSyn](https://github.com/amazon-science/tabsyn) codebase by adding:

- **TabKDE** methods
- **Hybrid models** that combine KDE and VAE/Diffusion approaches

---

## 📦 Dataset Preparation

1. **Download the Dataset**

   Run:
   ```bash
   python download_dataset.py
   ```
   *(For IBM dataset, use the link provided in the original paper.)*

2. **Process the Dataset**

   Run:
   ```bash
   python process_dataset.py
   ```

---

## 🧱 TabKDE / Simple KDE

**TabKDE:**
```bash
python main.py --dataname adult --method TabKDE --mode train
python main.py --dataname adult --method TabKDE --mode sample
```

**Simple_KDE (sampling only):**
```bash
python main.py --dataname adult --method simple_KDE --mode sample
```

---

## 🎯 Coreset Sampling

Requires **TabKDE** training first (uses copula embedding).

```bash
python coreset.py --dataname adult
python coreset_random.py --dataname adult
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
python main.py --dataname adult --method tabsyn --mode sample --latent_encoding copula_diff_encoding
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
python main.py --dataname "$dataname" --method baseline --mode train
```
