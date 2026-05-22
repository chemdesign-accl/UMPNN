# UMPNN: Undirected Message-Passing Neural Network

This folder contains the code and tabular data used for the UMPNN graph neural
network workflow developed for SARS-CoV-2 nsp13 inhibitor prediction.

The model represents each molecule as an undirected molecular graph and combines
atom-level, bond-level, and whole-molecule descriptor features to classify
compounds as active or inactive nsp13 inhibitor candidates.

## Folder Contents

| File | Description |
| --- | --- |
| `UMPNN.py` | Main Python script for training and evaluating the undirected message-passing neural network. The script reads `training_dataset.csv`, builds molecular graphs from SMILES strings, creates a Bemis-Murcko scaffold train/test split, trains the model, evaluates scaffold-test performance, and writes plots and diagnostic reports. |
| `training_dataset.csv` | Curated model dataset with 2,619 molecules. It contains compound identifiers, SMILES, binary activity labels, activity values, activity uncertainty, and activity-source metadata. |
| `total_tested.csv` | Experimental testing table for 75 selected compounds. It includes molecule IDs, screening round, SMILES, selectivity index, physicochemical properties, A549-hACE2 cytotoxicity, CPE EC50, nLuc EC50, nsp13 unwinding EC50, and Vero76 cytotoxicity results. |
| `Simulations.zip` | Molecular dynamics simulation archive. Include this archive in the upload when sharing the simulation setup, analysis scripts, and representative outputs described below. |

When `UMPNN.py` is run, it creates a `Plots/` directory containing training
metadata, scaffold-test plots, similarity diagnostics, and model evaluation
reports.

## Model Overview

UMPNN uses a graph-network style architecture implemented with PyTorch
Geometric's `MetaLayer`. The model contains three message-passing blocks:

- `EdgeBlock`: updates bond features using source atom, destination atom, and
  current edge attributes.
- `NodeBlock`: updates atom features by aggregating incoming edge messages.
- `GlobalBlock`: pools atom features across the molecular graph and combines
  them with global molecular descriptors before binary classification.

The graph is treated as undirected by adding both directions for each covalent
bond.

## Molecular Features

The model uses three feature groups:

- Atom features: atomic number, explicit degree, aromaticity, explicit valence,
  and hybridization.
- Bond features: bond length, bond type, conjugation, and ring-size encoding.
- Global molecular descriptors: heavy atom count, ring count, stereocenter
  count, hydrogen-bond donors, rotatable bonds, TPSA, logP, fraction sp3
  carbons, and radius of gyration.

RDKit is used for molecule parsing, descriptor calculation, conformer
generation, and Bemis-Murcko scaffold extraction.

## Dataset Columns

### `training_dataset.csv`

Important columns include:

- `Title`: compound identifier used by the model.
- `AviDD ID`: AViDD/MWAC identifier when available.
- `SMILES`: molecular structure used to construct the graph.
- `Label`: binary activity label, where `1` is active and `0` is inactive.
- `Activity_Value`, `Activity_Units`, `Activity_SD`, and `Activity_SE`:
  activity measurement and uncertainty metadata.
- `Activity_Source`: source or assay category used to define the activity
  measurement.

Activity classes were binarized using assay-specific thresholds. Cell-based
activity values were labeled active below 15 µM. HTS measurements were labeled
active above the 22.29 ± SE threshold.

### `total_tested.csv`

Important columns include:

- `Molecule`: tested molecule identifier.
- `Round`: experimental selection/testing round.
- `SMILES`: molecular structure.
- `A549-hACE2_Cytotoxicity: CC50 (µM)`: A549-hACE2 cytotoxicity.
- `CPE EC50 (µM)`: cytopathic-effect protection potency.
- `nLuc EC50 (µM)`: nanoluciferase assay potency.
- `P5_SARS-CoV2_nsp13-dsRNA unwinding: EC50 (µM)`: biochemical nsp13
  unwinding assay potency.
- `P5_Vero76_Cytotoxicity: CC50 (µM)`: Vero76 cytotoxicity.

Some experimental entries use text such as `> 50.00` or
`(CC50 could not be calculated)` to record assay limits or unavailable values.

## Main Parameters

The current script configuration is defined near the top of `UMPNN.py`:

- Hidden dimension: `400`
- Batch size: `248`
- Epochs: `50`
- Learning rate: `3.28e-4`
- Weight decay: `3.29e-2`
- Dropout: `0.25`
- Scaffold test fraction: `0.20`
- Classification threshold: `0`
- Cross-validation setting retained in the trainer: `5` folds

## Outputs

Running the script writes outputs to `Plots/`, including:

- `hyperparameters.json`
- `logfile.log`
- `conf_matrix.png`
- `roc_auc_scaffold_test_set.png`
- `train_test_tanimoto_summary.json`
- Scaffold-test train/evaluation Tanimoto overlap CSV files
- Similarity-bin and similarity-threshold performance CSV files
- Active/decoy similarity reports and histograms

The script also writes `3_metrics.png` and `validation_roc_plot.png` if the
corresponding plotting methods are called.

## Simulation Files

The archive `Simulations.zip` contains the input files, scripts, and
representative outputs required to reproduce the molecular dynamics simulations
and analyses reported in this work.

Contents:

- `System-prep-for-sim/`: directory containing the parameter files and initial
  coordinate files used to build the simulation systems. It also includes the
  preparation workflow used to generate the final simulation systems starting
  from the docked poses.
- `DBSCAN-clustering.in`: input script for CPPTRAJ used to perform DBSCAN
  hierarchical clustering in order to classify the dominant conformational
  states observed during the simulations.
- `DBSCAN_representative_39XX.pdb`: representative structures obtained from the
  DBSCAN clustering analysis for the two primary hit compounds identified in
  this study.
- `FINAL_DECOMP_MMGBSA_39XX.dat`: per-residue MM/GBSA energy decomposition
  results derived from the analyzed trajectories of the two hit systems.
- `mmpbsa.in` and `mmpbsa_recipe.txt`: input configuration and protocol used to
  perform MM/GBSA calculations in AMBER, following the procedure described in
  `mmpbsa_recipe.txt`.
- `contacts.py`: Python script used to extract protein-ligand contacts within a
  5.5 Angstrom cutoff around the ligand throughout the simulation trajectories.

## Dependencies

The workflow requires Python and the following packages:

- `torch`
- `torch-geometric`
- `torch-scatter`
- `rdkit`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tqdm`

Install PyTorch, PyTorch Geometric, and `torch-scatter` using versions that are
compatible with the available CUDA or CPU environment.

## Usage

Run from inside this folder:

```bash
python UMPNN.py
```

The script expects `training_dataset.csv` to be present in the same directory. It
will create `Plots/` automatically.

## Notes

- The scaffold split is generated at runtime from `training_dataset.csv`.
- Molecules sharing the same Bemis-Murcko scaffold are kept in the same split.
- The model is trained on the scaffold-training subset and evaluated on the
  held-out scaffold-test subset.
- `total_tested.csv` is included as the experimental follow-up testing table and
  is not required for training the model.
