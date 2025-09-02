# SHERPA: SHAP-based Explainable Robust Poisoning Aggregation for Federated Learning

**SHERPA** is an explainability-driven defence against **data poisoning attacks** in Federated Learning (FL).  
It computes **SHAP** feature attributions over client updates/models, **clusters** those attribution patterns (e.g., via **HDBSCAN**) to detect anomalies, and **excludes or down-weights** suspicious clients before aggregation—while keeping decisions **interpretable**.

> **Original work (please cite):**  
> Sandeepa, C., Siniarski, B., Wang, S. and Liyanage, M., 2024, May. *SHERPA: Explainable robust algorithms for privacy-preserved federated learning in future networks to defend against data poisoning attacks.* In **2024 IEEE Symposium on Security and Privacy (SP)** (pp. 4772–4790). IEEE.

---

## 📂 Repository Layout

```text
SHERPA/
├─ notebooks/
│  └─ sherpa_demo.ipynb                    # End-to-end demo & visualization
│
├─ src/
│  ├─ dataset/
│  │  ├─ customDatasets/
│  │  │  ├─ CelebA.py                      # CelebA dataset wrapper
│  │  │  ├─ CustomDataset.py               # Base dataset abstraction
│  │  │  ├─ NIDD_5G.py                     # 5G-NIDD dataset wrapper
│  │  │  └─ NSL_KDD.py                     # NSL-KDD dataset wrapper
│  │  ├─ dataLoaderFactory.py              # Factory for DataLoaders
│  │  ├─ datasetHandler.py                 # Splits, transforms
│  │  ├─ datasetStrategy.py                # Dataset selection strategies
│  │  └─ poisoning.py                      # (Optional) data-poison helpers
│  │
│  ├─ FLProcess/
│  │  ├─ CustomAggregate.py                # Aggregation entry (hooks SHERPA)
│  │  ├─ CustomFedAvg.py                   # FedAvg implementation
│  │  ├─ CustomFedAvgBase.py               # Common FedAvg helpers
│  │  ├─ CustomFLAME.py                    # FLAME baseline (if enabled)
│  │  ├─ CustomKrum.py                     # Krum baseline
│  │  ├─ DPFlowerClient.py                 # (Optional) DP-enabled Flower client
│  │  ├─ FlowerClient.py                   # Flower client abstraction
│  │  └─ FLUtil.py                         # FL utilities (sampling, metrics)
│  │
│  ├─ NN/
│  │  ├─ MdlTraining.py                    # Train/eval loops
│  │  ├─ NNConfig.py                       # Model/training config
│  │  ├─ NNUtil.py                         # Save/load, seeds, helpers
│  │  └─ ResNet.py                         # Example model(s)
│  │
│  ├─ poisonDetection/
│  │  ├─ clientAnalysis/
│  │  │  ├─ strategyFnDebugging.py         # Debug hooks for strategies
│  │  │  ├─ strategyFnGeneralAlg.py        # General poisoning baseline
│  │  │  └─ strategyFnRandomPoison.py      # Random poison baseline
│  │  ├─ clusteringHDBSCAN.py              # HDBSCAN clustering of SHAP
│  │  ├─ tsneVisualisation.py              # t-SNE plots of attribution space
│  │  └─ xaiMetrics.py                     # SHAP computation & XAI metrics
│  │
│  └─ util/
│     └─ constants.py                      # Paths, seeds, hyperparams
│
├─ main.py                                  # Scripted entry-point (experiments)
├─ .gitignore
└─ README.md

## SHERPA Summary

1. **Explain** client updates/models with **SHAP** (e.g., class-conditional, layer/feature-wise).
2. **Embed & Cluster** the attribution vectors across clients (**HDBSCAN** by default).
3. **Detect anomalies**: sparse/low-density clusters & outliers → **suspect poisoners**.
4. **Defend**: drop or **down-weight** suspects before aggregation (**FedAvg / Krum / FLAME**).
5. **Interpret**: store **reason codes** (attribution patterns/cluster labels) for **auditability**.

---

## Features

- **Explainable** robust FL via **SHAP + density-based clustering**.  
- Multiple robust **baselines**: **FedAvg**, **Krum**, **FLAME** (and optional **DP**).  
- **Dataset adapters**: **NSL-KDD**, **5G-NIDD**, **CelebA** (easy to extend).  
- **Visual analytics**: **t-SNE** of attribution space, cluster summaries.  
- **Pluggable strategies** for attack/defence with minimal glue code.
