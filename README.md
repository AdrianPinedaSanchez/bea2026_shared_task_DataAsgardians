# Data Asgardians at BEA 2026 Shared Task 1

## A Hybrid Transformer–Feature Ensemble for L1-Aware English Vocabulary Difficulty Prediction

**Authors:** Adrian Pineda, Sabur Butt, and Héctor Ceballos Cancino

**Published in:** Proceedings of the 21st Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2026), Association for Computational Linguistics, pages 1137–1145. San Diego, California, USA.

**Paper:** https://aclanthology.org/2026.bea-1.82/

This repository is the reproducibility record for team Data Asgardians' submission to Shared Task 1 of BEA 2026, on predicting the Knowledge of Vocabulary Level (KVL) of English words for learners from three first-language (L1) backgrounds: Spanish, German, and Mandarin Chinese. It contains the complete research pipeline behind the paper cited above, from exploratory data analysis and linguistic feature engineering to transformer fine-tuning and the final submitted systems. All notebooks are provided with their stored outputs so that the full experimental record can be inspected without access to a GPU.

Our approach combines multilingual transformer models (XLM-RoBERTa-Large and mDeBERTa-v3-base) with a set of theoretically motivated linguistic features covering phonology, morphology, orthography, and lexical semantics. The best submitted system, an ensemble built on mDeBERTa-v3-base, reached a test RMSE of 1.037 (Spanish), 0.997 (German), and 0.913 (Mandarin), improving on the official xlm-roberta-base baseline for every L1.

## 1. Task Description

The shared task is organized by the British Council on data described in Skidmore et al. (2025) and grounded in the vocabulary knowledge framework of Schmitt et al. (2024). Each item pairs an English target word with its part of speech, a short English clue sentence, the corresponding word in the learner's L1, and an L1 context sentence. The prediction target is `GLMM_score`, a continuous difficulty estimate obtained from a generalized linear mixed model fitted to learner responses. Higher scores indicate easier words and lower scores indicate harder words.

Two evaluation tracks are defined:

- **Closed track**: one model per L1, trained only on that L1's data.
- **Open track**: any training configuration is allowed, including joint training across L1s.

Systems are evaluated per L1 with root mean squared error (RMSE) and Pearson correlation against the held-out labels.

The official baseline code and data are distributed by the organizers at:

> https://github.com/britishcouncil/bea2026st

The data are not redistributed in this repository. Section 6 explains how to obtain them.

## 2. Repository Map

| Path | Content |
|------|---------|
| `README.md` | This document |
| `paper/Paper.pdf` | The system description paper |
| `notebooks/01_eda_complete.ipynb` | Exploratory data analysis and feature engineering (CPU only) |
| `notebooks/02_phase1_model_improvement.ipynb` | Phase 1 experiments: XLM-RoBERTa-Large with linguistic features |
| `notebooks/03_transformer_architecture_comparison.ipynb` | Comparative evaluation of transformer backbones |
| `notebooks/04_final_pipeline.ipynb` | End-to-end pipeline that produced the submitted systems |
| `scripts/regen_eda_figs.py` | Regenerates the EDA figures used in the paper (CPU only) |
| `scripts/regen_fig12.py` | Redraws the final results figure from the recorded test RMSE values |
| `scripts/results_summary.py` | Helper that generates the interim results comparison cell |
| `figures/eda/` | Original exploratory analysis figures produced by notebook 01 |
| `figures/paper/` | Publication figures: recolored EDA figures, SHAP summaries, and the final results comparison |

## 3. Research Pipeline

The work proceeded in four stages, and the notebook numbering reflects that chronology. Each notebook opens with a header cell stating its purpose, its hardware requirements, and its place in the pipeline.

### Stage 1: Exploratory Data Analysis (`01_eda_complete.ipynb`)

This notebook characterizes the dataset and motivates the modeling decisions. It engineers 22 linguistic features spanning orthography (word length, clue ratio), phonology from the CMU Pronouncing Dictionary (phoneme counts, silent letters, consonant clusters), L1-specific phonology informed by PHOIBLE (phonemes absent from the learner's L1, unfamiliar letters), lexical semantics from WordNet (polysemy, homonymy, approximate frequency), cross-lingual form distance (character n-gram cosine distance and Levenshtein distance between the English word and its L1 translation), and a heuristic measure of morphological complexity.

The analysis establishes several findings that shaped the later stages. Word frequency and cross-lingual form distance are the strongest individual correlates of difficulty. Several features behave differently across L1s; for example, phoneme-level difficulty matters more for Mandarin speakers than for German speakers, which supports per-L1 modeling in the closed track. A number of feature pairs are highly collinear, and an automatic selection step with a threshold of |r| > 0.85 reduces the feature set without losing signal.

### Stage 2: Phase 1 Modeling (`02_phase1_model_improvement.ipynb`)

This notebook isolates a single question: how much do the linguistic features help when combined with one main model, XLM-RoBERTa-Large? Three levels of evidence are examined in sequence. First, feature diagnostics verify that the linguistic variables carry independent signal. Second, a tabular baseline (random forest and gradient boosting on features alone) quantifies how much the features explain by themselves. Third, two integration methods are tested: serializing the features as short text tags prepended to the transformer input, and a hybrid method that feeds frozen XLM-RoBERTa-Large embeddings, reduced to 128 dimensions with PCA, together with the tabular features into an XGBoost regressor.

### Stage 3: Architecture Comparison (`03_transformer_architecture_comparison.ipynb`)

A compact harness for comparing larger transformer backbones under both the closed setting (one model per L1) and the open setting (a single joint model). Its main output is a comparative table of RMSE and Pearson correlation per model and scenario, which informed the selection of XLM-RoBERTa-Large and mDeBERTa-v3-base for the final systems.

### Stage 4: Final Pipeline (`04_final_pipeline.ipynb`)

The end-to-end pipeline behind the submission. It trains and evaluates three systems: Closed Solo (XLM-RoBERTa-Large fine-tuned per L1 without features), Closed Hybrid (XLM-RoBERTa-Large combined with the linguistic features through an MLP head, warm-started from the Solo weights), and Open (a single XLM-RoBERTa-Large fine-tuned jointly on all three L1s). It also trains per-L1 XGBoost models with Optuna hyperparameter search as a tabular reference. After development-set evaluation, every model is retrained on the union of the training and development sets before producing the final test predictions and the submission archive. Transformer checkpoints (about 1.3 GB each) are persisted to Google Drive; predictions, XGBoost models, and result tables are versioned in Git.

## 4. Results

Official baseline results on the development set (xlm-roberta-base, open track), as distributed with the organizers' repository:

| L1 | RMSE | Pearson |
|----|------|---------|
| Spanish (ES) | 1.206 | 0.787 |
| German (DE) | 1.149 | 0.800 |
| Mandarin (CN) | 1.021 | 0.804 |

Test-set RMSE of all systems evaluated in this work, as reported in the paper and in `figures/paper/fig12_test_rmse_per_l1_all_models.png` (lower is better):

| System | ES | DE | CN |
|--------|------|------|------|
| xlm-roberta-base (baseline) | 1.257 | 1.258 | 1.140 |
| xlm-roberta-large, Solo | 1.182 | 1.177 | 1.008 |
| xlm-roberta-large, Hybrid | 1.186 | 1.117 | 1.006 |
| XGBoost (features only) | 1.461 | 1.351 | 1.279 |
| Ridge (features only) | 1.505 | 1.407 | 1.293 |
| mDeBERTa-v3-base, Solo | 1.152 | 1.141 | 1.037 |
| mDeBERTa-v3-base, Hybrid | 1.180 | 1.157 | 1.112 |
| mDeBERTa-v3-base, Ensemble | 1.037 | 0.997 | 0.913 |

Three observations summarize the outcome. Purely tabular models trained on the linguistic features alone fall well short of the transformer systems, confirming that contextual representations carry most of the predictive signal. The linguistic features nevertheless contribute: the Hybrid variant of XLM-RoBERTa-Large improves over its Solo counterpart for German and Mandarin. The strongest system for every L1 is the mDeBERTa-v3-base Ensemble, which reduces test RMSE relative to the official baseline by 17.5 percent for Spanish, 20.7 percent for German, and 19.9 percent for Mandarin.

## 5. Hardware and Execution Record

All transformer training was carried out on Google Colab Pro with dedicated GPUs, specifically an NVIDIA T4 for the Phase 1 experiments and an NVIDIA A100 for the larger comparative and final runs. The notebooks in this repository are published with the outputs of those runs preserved, which serves two purposes: the complete experimental record can be audited cell by cell without re-execution, and readers without GPU access can still follow every result.

Approximate execution times, for reference:

- Notebook 01 (EDA): under 10 minutes on CPU.
- Notebook 02 (Phase 1): about 15 minutes for the fine-tuned transformer and 5 minutes for embedding extraction on a T4.
- Notebook 04 (final pipeline): about 40 minutes for a full training pass; about 2 minutes when loading existing checkpoints from Google Drive.

## 6. Reproducing the Work

### 6.1 Obtaining the data

The task data belong to the British Council and are obtained from the official repository:

```bash
git clone https://github.com/AdrianPinedaSanchez/bea2026_shared_task_DataAsgardians.git
cd bea2026_shared_task_DataAsgardians
git clone https://github.com/britishcouncil/bea2026st.git
```

This places the expected directory layout, `bea2026st/data/{train,dev,test}/{es,de,cn}/`, at the repository root. The notebooks and the figure scripts read the CSV files from that location.

### 6.2 Environment

The CPU-side analysis requires Python 3.10 or later with numpy, pandas, matplotlib, seaborn, scipy, scikit-learn, xgboost, lightgbm, optuna, nltk, torch, and transformers. The NLTK resources wordnet, omw-1.4, and cmudict are downloaded automatically by the scripts on first use. For GPU training, the notebooks were executed on Google Colab, where the setup cells install the required versions.

### 6.3 Inspecting results without a GPU

Open the notebooks directly on GitHub or in Jupyter. All tables, metrics, and figures produced during the original runs are stored in the notebooks. This is the recommended path for verification of the reported results.

### 6.4 Regenerating the paper figures

Both figure scripts run on CPU:

```bash
python scripts/regen_eda_figs.py
python scripts/regen_fig12.py
```

The first script recomputes the full feature set from the raw training data and redraws the target distribution and feature correlation figures. The second script redraws the final results comparison; note that it plots the recorded test RMSE values of the completed GPU runs rather than retraining any model, since retraining requires the GPU pipeline of notebook 04.

### 6.5 Re-running the training

Re-executing notebooks 02, 03, and 04 requires a CUDA GPU with at least 16 GB of memory. Each notebook states its own requirements in the header cell. Notebook 04 additionally uses Google Drive for checkpoint persistence and detects existing checkpoints automatically, so an interrupted run can be resumed without repeating completed training.

## 7. Figures

The `figures/eda/` directory holds the ten figures produced by the exploratory analysis in notebook 01, covering the target distribution, feature distributions, feature correlations with the target, multicollinearity, per-L1 feature importance, cross-L1 comparisons, and part-of-speech effects.

The `figures/paper/` directory holds the publication versions: the recolored target distribution and feature correlation figures, SHAP summary plots for the per-L1 XGBoost models (individually and combined), and the final test RMSE comparison across all eight systems (Figure 12), in both PNG and PDF form.

## 8. Citation

If you use this work, please cite the paper:

```bibtex
@inproceedings{pineda-etal-2026-data,
    title = "Data Asgardians at {BEA} 2026 Shared Task 1: A Hybrid Transformer{--}Feature Ensemble for {L}1-Aware {E}nglish Vocabulary Difficulty Prediction",
    author = "Pineda, Adrian and Butt, Sabur and Ceballos Cancino, H{\'e}ctor",
    booktitle = "Proceedings of the 21st Workshop on Innovative Use of {NLP} for Building Educational Applications ({BEA} 2026)",
    month = jul,
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    pages = "1137--1145",
    doi = "10.18653/v1/2026.bea-1.82"
}
```

## 9. References

- Skidmore et al. (2025). Dataset description underlying the BEA 2026 KVL shared task.
- Schmitt, N., et al. (2024). Knowledge of Vocabulary Level framework.
- Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. XLM-RoBERTa.
- He, P., et al. (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training. mDeBERTa-v3.

## 10. License and Acknowledgements

The code in this repository is released under the terms of the LICENSE file. The shared task data remain the property of the British Council and are subject to the terms stated in the official task repository. We thank the shared task organizers for providing the data, the baseline systems, and the evaluation infrastructure.
