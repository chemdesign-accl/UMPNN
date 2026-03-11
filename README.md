<div align="center" style="font-size: 22px; line-height: 1.25;">
<pre>
                         Atom i  ● <───────── ⇄ ─────────> ●  Atom j                          
                          \                      /                          
● ──── ⌬ ─── ● ──── ⌬ ─── ●        UMPNN        ● ──── ⌬ ─── ● ──── ⌬ ─── ●
                          /                      \                          
                         Atom j  ● <───────── ⇄ ─────────> ●  Atom i                          

</pre>

</div>

# **UMPNN: Undirected Message-Passing Neural Network**

### Architecture used for Molecular Prediction of Nsp13 Inhibition

This repository contains a Message-Passing Neural Network (MPNN) used to classify small molecules as **active** or **inactive** inhibitors of the SARS-CoV-2 helicase Nsp13. The model operates directly on molecular graphs using a symmetric adjacency matrix to create undirection and integrates **atom-, bond-, and global-level features** to capture both local chemistry and overall drug-like properties.

This architecture was coupled to a stucture-based funnel-like virtual screening protocol and allowed us to identify two compounds with IC50 < 600 nm and 17 compounds with IC50 < 10 μM in cell-based nLuc essays, , underscoring the strong predictive power and practical utility of the workflow.

---

## 1. Method Overview

### Training Data

- **Curated labeled dataset:** 2,700 compounds  
  - Primary labels from **titration assays** measuring Nsp13 unwinding inhibition.  
  - Augmented with a **small subset of nanoluciferase (nLuc) cell-based EC₅₀ values** used to refine activity labels.
- **Binary labels:**  
  - 1 = active (EC₅₀ below activity threshold set to 15 μM)  
  - 0 = inactive / decoy  

A **blind test set comprising 20% of the data** was constructed by **stratified sampling**, preserving the class balance of actives and inactives.

### Model Architecture

The model is implemented as a **three-block MetaLayer**[1] :

- **EdgeBlock:** updates bond features from source/destination node features and edge attributes.
- **NodeBlock:** updates node features by aggregating incoming edge messages via `scatter(..., reduce='max')`.
- **GlobalBlock:** performs `global_max_pool` over node features and concatenates them with **global molecular descriptors**, followed by an MLP that outputs a single logit (binary classification).

All blocks use **Batch Normalization**, **ReLU**, and **Dropout** to improve stability and generalization.

Key hyperparameters (current configuration):

- Hidden dimension: `400`
- Batch size: `248`
- Epochs: `50`
- Learning rate: `3.28e-4`
- Weight decay: `3.29e-2`
- Dropout: `0.25`
- Loss: `BCEWithLogitsLoss`
- Optimizer: `Adam`
- 5-fold cross-validation
- 
[1] Peter W. Battaglia, Jessica B. Hamrick, et. al., Relational inductive biases, deep learning, and graph networks, 2018. URL https://arxiv.org/abs/1806.01261

---

## 2. Molecular Features

The network uses **three levels of features**:

### Global Features (per-molecule)

Global descriptors are computed with RDKit and **min–max normalized** (or scaled as indicated). Only weakly correlated properties (pairwise |r| < 0.85) were retained to provide complementary information:

1. Number of heavy atoms  
2. Ring count (scaled by `/6`)  
3. Number of stereocenters (scaled by `/5`)  
4. Number of H-bond donors  
5. Number of rotatable bonds  
6. TPSA  
7. logP  
8. Fraction sp³ carbons  
9. Radius of gyration (3D compactness)

### Atom Features

For each non-hydrogen atom:

- Min–max normalized **atomic number**
- Scaled **explicit degree** (value/3)
- Aromaticity (0 = no, 1 = yes)
- Scaled explicit valence (value/4)
- Hybridization encoded as:  
  - sp = 0, sp² = 0.5, sp³ = 1

### Bond Features

For each covalent bond (duplicated A→B and B→A):

- Min–max normalized **bond length**
- Bond type: single = 1, double = 2, aromatic = 1.5 (then scaled)
- Conjugation state (0 = no, 1 = yes)
- Ring size feature: encoded from 0 (no ring) to 1 (large/other rings), with specific values for 4–8-membered rings.

---

## 3. Model Performance

On the **stratified blind test set (20% of the curated dataset)**, the MPNN achieved:

- **Accuracy:** 91.0 ± 1.2%  
- **F1-score:** 84 ± 2.3%  
- **AUC (ROC):** 0.962 ± 0.2  

These metrics demonstrate that the graph neural network can reliably distinguish **active vs inactive Nsp13 inhibitors** based on molecular structure alone.

The trained model was subsequently applied to a virtual library of **1,172,240 commercial drug-like compounds**, from which it identified **1,866 candidate inhibitors** for further docking and MD-based evaluation.

---

## 4. Repository Contents

- `UMPNN.py` – main training and evaluation script implementing:
  - `MolecularDataset` for SMILES → graph conversion
  - `InteractionNetwork` (EdgeBlock, NodeBlock, GlobalBlock)
  - `GNNTrainer` class for k-fold cross-validation, training, and evaluation
  - Blind-set evaluation and plotting (confusion matrix, ROC curve)
- `combined_decoy_EC50_cleaned.csv` – curated training dataset containing:  
  - `SMILES`, `Title`, `Actividad` (binary label)
- `blind_compounds.csv` – stratified blind test set with the same columns.
- 'total_tested.csv' - Experimentally tested results for the 75 total compounds discover in our MidWest AviDD SARS-COV2 Nsp13 inhibitor campaign.

---

## 5. Dependencies

The code is written in Python and uses:

- `python >= 3.8`
- `pytorch`
- `torch-geometric`
- `torch-scatter`
- `rdkit`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tqdm`

Install with `conda` or `pip` as appropriate, making sure that your `torch-geometric` version matches the installed PyTorch and CUDA.

---

## 6. Usage for blind test prediction

1. Place the CSV files in the working directory:

   - `combined_decoy_EC50_cleaned.csv`
   - `blind_compounds.csv`

   Each must contain at least:

   - `SMILES` – canonical SMILES string  
   - `Title` – compound identifier  
   - `Actividad` – 0 (inactive) or 1 (active)

2. Run the training and blind-set evaluation:

   ```bash
   python UMPNN.py


