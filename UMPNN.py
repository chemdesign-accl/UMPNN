#!/usr/bin/env python
# coding: utf-8
import logging
import json
import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, MetaLayer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_scatter import scatter
from sklearn.model_selection import KFold
import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import WeightedRandomSampler
import os
from rdkit import Chem
from rdkit import RDLogger
from rdkit import DataStructs
from rdkit.Chem import (
    AllChem, Draw, Descriptors)
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')   # completely suppress RDKit warnings
input_dir = "./Plots"
os.makedirs(input_dir, exist_ok=True)
#Parameters
hidden_channels= 400
lr= 0.00032808125988928395
weight_decay= 0.03291489371381885
output=2
batch_size= 248
k_folds= 5
epochs= 50
dropout_rate= 0.25
threshold = 0
scaffold_test_fraction = 0.20
blind_test_path = "nsp13_legacy_stratified_blind_labels.csv"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
edge_dim = 4
global_dim = 9
num_node_features = 5
tanimoto_radius = 2
tanimoto_n_bits = 2048
# Set up logging
log_file_path = os.path.join(input_dir, "logfile.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path, 
    filemode='w' 
)
logger = logging.getLogger(__name__)


def report_hyperparameters():
    hyperparameters = {
        "hidden_channels": hidden_channels,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "k_folds": k_folds,
        "epochs": epochs,
        "dropout_rate": dropout_rate,
        "classification_threshold": threshold,
        "edge_dim": edge_dim,
        "global_dim": global_dim,
        "num_node_features": num_node_features,
        "scaffold_test_fraction": scaffold_test_fraction,
        "blind_test_path": blind_test_path,
        "tanimoto_radius": tanimoto_radius,
        "tanimoto_n_bits": tanimoto_n_bits,
        "device": str(device),
    }
    logger.info("Hyperparameters: %s", hyperparameters)
    with open(os.path.join(input_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f, indent=2)
    print("Hyperparameters used:")
    for name, value in hyperparameters.items():
        print(f"  {name}: {value}")
    return hyperparameters


def get_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(smiles_list, labels, titles, test_fraction=0.20, return_indices=False):
    scaffold_to_indices = {}
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_murcko_scaffold(smiles)
        if scaffold is None:
            scaffold = f"invalid_{idx}"
        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    scaffold_groups = sorted(
        scaffold_to_indices.values(),
        key=lambda group: (len(group), group[0]),
        reverse=True,
    )

    n_total = len(smiles_list)
    n_test_target = int(round(n_total * test_fraction))
    train_idx = []
    test_idx = []

    for group in scaffold_groups:
        if len(test_idx) + len(group) <= n_test_target or not test_idx:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    train_idx = sorted(train_idx)
    test_idx = sorted(test_idx)

    split_summary = {
        "n_total": n_total,
        "n_train": len(train_idx),
        "n_scaffold_test": len(test_idx),
        "n_scaffolds_total": len(scaffold_to_indices),
        "n_scaffolds_train": len({
            get_murcko_scaffold(smiles_list[i]) for i in train_idx
        }),
        "n_scaffolds_test": len({
            get_murcko_scaffold(smiles_list[i]) for i in test_idx
        }),
    }
    logger.info("Scaffold split summary: %s", split_summary)
    print("Scaffold split summary:")
    for name, value in split_summary.items():
        print(f"  {name}: {value}")

    split_result = (
        [smiles_list[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [titles[i] for i in train_idx],
        [smiles_list[i] for i in test_idx],
        [labels[i] for i in test_idx],
        [titles[i] for i in test_idx],
    )
    if return_indices:
        return split_result + (train_idx, test_idx)
    return split_result


def smiles_to_morgan_fp(smiles, radius=tanimoto_radius, n_bits=tanimoto_n_bits):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def canonical_smiles(smiles, include_chirality=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=include_chirality)


def join_unique(values):
    joined_values = []
    for value in values:
        if pd.isna(value):
            continue
        value = str(value).strip()
        if value and value.lower() != "nan" and value not in joined_values:
            joined_values.append(value)
    return "; ".join(joined_values)


def write_train_eval_tanimoto_overlap_report(train_df, eval_df, eval_name, output_dir=input_dir):
    """Write train-vs-evaluation nearest-neighbor Tanimoto diagnostics.

    Tanimoto = 1.0 means the Morgan bit vectors are identical. The extra
    canonical-SMILES columns distinguish true repeated structures from cases
    where different molecules collapse to the same fingerprint.

    Max_Train_Tanimoto is the nearest-neighbor similarity: for each evaluation
    molecule, it stores the highest Tanimoto similarity against any training
    molecule. Mean_Train_Tanimoto is the average similarity of that evaluation
    molecule against every training molecule, which is useful for measuring the
    global train/evaluation separation rather than just the closest analogue.
    """
    train_df = train_df.reset_index(drop=True).copy()
    eval_df = eval_df.reset_index(drop=True).copy()

    train_df["_canonical_smiles"] = train_df["SMILES"].map(lambda smiles: canonical_smiles(smiles, True))
    train_df["_canonical_smiles_no_stereo"] = train_df["SMILES"].map(lambda smiles: canonical_smiles(smiles, False))
    eval_df["_canonical_smiles"] = eval_df["SMILES"].map(lambda smiles: canonical_smiles(smiles, True))
    eval_df["_canonical_smiles_no_stereo"] = eval_df["SMILES"].map(lambda smiles: canonical_smiles(smiles, False))

    train_fps = [smiles_to_morgan_fp(smiles) for smiles in train_df["SMILES"]]
    eval_fps = [smiles_to_morgan_fp(smiles) for smiles in eval_df["SMILES"]]
    valid_train_idx = [idx for idx, fp in enumerate(train_fps) if fp is not None]
    valid_train_fps = [train_fps[idx] for idx in valid_train_idx]

    rows = []
    for eval_idx, eval_row in eval_df.iterrows():
        eval_fp = eval_fps[eval_idx]
        if eval_fp is None or not valid_train_fps:
            max_tanimoto = np.nan
            mean_tanimoto = np.nan
            tanimoto_1_train_idx = []
        else:
            similarities = DataStructs.BulkTanimotoSimilarity(eval_fp, valid_train_fps)
            max_tanimoto = max(similarities) if similarities else np.nan
            mean_tanimoto = float(np.mean(similarities)) if similarities else np.nan
            tanimoto_1_train_idx = [
                valid_train_idx[idx]
                for idx, similarity in enumerate(similarities)
                if np.isclose(similarity, 1.0)
            ]

        exact_isomeric_idx = [
            idx for idx in tanimoto_1_train_idx
            if train_df.loc[idx, "_canonical_smiles"] == eval_row["_canonical_smiles"]
        ]
        exact_nonisomeric_idx = [
            idx for idx in tanimoto_1_train_idx
            if train_df.loc[idx, "_canonical_smiles_no_stereo"] == eval_row["_canonical_smiles_no_stereo"]
        ]
        matched_train_idx = exact_isomeric_idx or exact_nonisomeric_idx or tanimoto_1_train_idx
        matched_train = train_df.loc[matched_train_idx].copy() if matched_train_idx else pd.DataFrame()

        rows.append({
            "Max_Train_Tanimoto": max_tanimoto,
            "Mean_Train_Tanimoto": mean_tanimoto,
            "Tanimoto_1_Train_Match_Count": len(tanimoto_1_train_idx),
            "Exact_Isomeric_Train_Match_Count": len(exact_isomeric_idx),
            "Exact_NonIsomeric_Train_Match_Count": len(exact_nonisomeric_idx),
            "Tanimoto_1_Is_Exact_Isomeric_Structure": len(exact_isomeric_idx) > 0,
            "Tanimoto_1_Is_Exact_NonIsomeric_Structure": len(exact_nonisomeric_idx) > 0,
            "Matched_Train_Title": join_unique(matched_train["Title"]) if not matched_train.empty else "",
            "Matched_Train_AviDD_ID": join_unique(matched_train["AviDD ID"]) if not matched_train.empty and "AviDD ID" in matched_train else "",
            "Matched_Train_SMILES": join_unique(matched_train["SMILES"]) if not matched_train.empty else "",
            "Matched_Train_Label": join_unique(matched_train["Label"]) if not matched_train.empty and "Label" in matched_train else "",
            "Matched_Train_Activity_Value": join_unique(matched_train["Activity_Value "]) if not matched_train.empty and "Activity_Value " in matched_train else "",
            "Matched_Train_Activity_Units": join_unique(matched_train["Activity_Units"]) if not matched_train.empty and "Activity_Units" in matched_train else "",
            "Matched_Train_Activity_SD": join_unique(matched_train["Activity_SD"]) if not matched_train.empty and "Activity_SD" in matched_train else "",
            "Matched_Train_Activity_SE": join_unique(matched_train["Activity_SE"]) if not matched_train.empty and "Activity_SE" in matched_train else "",
            "Matched_Train_Activity_Source": join_unique(matched_train["Activity_Source"]) if not matched_train.empty and "Activity_Source" in matched_train else "",
        })

    overlap = pd.DataFrame(rows)
    annotated_eval = pd.concat(
        [eval_df.drop(columns=["_canonical_smiles", "_canonical_smiles_no_stereo"]), overlap],
        axis=1,
    )
    tanimoto_1_matches = annotated_eval[
        np.isclose(annotated_eval["Max_Train_Tanimoto"], 1.0, equal_nan=False)
    ].copy()

    full_path = os.path.join(output_dir, f"{eval_name}_train_tanimoto_overlap.csv")
    matches_path = os.path.join(output_dir, f"{eval_name}_train_tanimoto_1_matches.csv")
    summary_path = os.path.join(output_dir, f"{eval_name}_train_tanimoto_overlap_summary.json")

    annotated_eval.to_csv(full_path, index=False)
    tanimoto_1_matches.to_csv(matches_path, index=False)

    summary = {
        "evaluation_set": eval_name,
        "n_train": int(len(train_df)),
        "n_evaluation": int(len(eval_df)),
        "n_tanimoto_1": int(len(tanimoto_1_matches)),
        "n_exact_isomeric_duplicates": int(tanimoto_1_matches["Tanimoto_1_Is_Exact_Isomeric_Structure"].sum()),
        "n_exact_nonisomeric_duplicates": int(tanimoto_1_matches["Tanimoto_1_Is_Exact_NonIsomeric_Structure"].sum()),
        "mean_pairwise_train_eval_tanimoto": float(annotated_eval["Mean_Train_Tanimoto"].mean()),
        "output_csv": full_path,
        "tanimoto_1_matches_csv": matches_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{eval_name} train/evaluation Tanimoto overlap summary:")
    for name, value in summary.items():
        print(f"  {name}: {value}")
    logger.info("%s train/evaluation Tanimoto overlap summary: %s", eval_name, summary)
    return annotated_eval, tanimoto_1_matches, summary


def classification_metrics_for_subset(y_true, y_score, y_pred):
    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "auc_roc": np.nan,
        }

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": np.nan,
    }
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        metrics["auc_roc"] = auc(fpr, tpr)
    return metrics


def write_similarity_performance_report(
    overlap_df,
    compound_titles,
    true_labels,
    prediction_scores,
    predicted_labels,
    eval_name,
    thresholds=(0.25, 0.50, 0.75, 0.90),
    output_dir=input_dir,
):
    prediction_df = pd.DataFrame({
        "Title": compound_titles,
        "True_Label": np.asarray(true_labels, dtype=int),
        "Prediction_Score": np.asarray(prediction_scores, dtype=float),
        "Prediction_Probability": 1 / (1 + np.exp(-np.asarray(prediction_scores, dtype=float))),
        "Predicted_Label": np.asarray(predicted_labels, dtype=int),
    })

    prediction_overlap = overlap_df.merge(prediction_df, on="Title", how="inner")
    if len(prediction_overlap) != len(prediction_df):
        logger.warning(
            "%s similarity performance matched %d predictions to %d overlap rows.",
            eval_name,
            len(prediction_df),
            len(prediction_overlap),
        )

    total_predictions = len(prediction_overlap)
    threshold_rows = []
    for threshold_value in thresholds:
        subset = prediction_overlap[prediction_overlap["Max_Train_Tanimoto"] >= threshold_value]
        y_true = subset["True_Label"].to_numpy(dtype=int)
        y_score = subset["Prediction_Score"].to_numpy(dtype=float)
        y_pred = subset["Predicted_Label"].to_numpy(dtype=int)
        metrics = classification_metrics_for_subset(y_true, y_score, y_pred)
        threshold_rows.append({
            "similarity_group": f">= {threshold_value:.2f}",
            "threshold_type": "cumulative",
            "threshold": threshold_value,
            "n_compounds": int(len(subset)),
            "fraction_of_evaluation_set": float(len(subset) / total_predictions) if total_predictions else np.nan,
            "n_positive": int(np.sum(y_true == 1)) if len(y_true) else 0,
            "n_negative": int(np.sum(y_true == 0)) if len(y_true) else 0,
            **metrics,
        })

    bin_edges = [-np.inf, 0.25, 0.50, 0.75, np.inf]
    bin_labels = ["< 0.25", "0.25-<0.50", "0.50-<0.75", "0.75-1.00"]
    prediction_overlap["Similarity_Bin"] = pd.cut(
        prediction_overlap["Max_Train_Tanimoto"],
        bins=bin_edges,
        labels=bin_labels,
        right=False,
    )

    bin_rows = []
    for bin_label in bin_labels:
        subset = prediction_overlap[prediction_overlap["Similarity_Bin"].astype(str) == bin_label]
        y_true = subset["True_Label"].to_numpy(dtype=int)
        y_score = subset["Prediction_Score"].to_numpy(dtype=float)
        y_pred = subset["Predicted_Label"].to_numpy(dtype=int)
        metrics = classification_metrics_for_subset(y_true, y_score, y_pred)
        bin_rows.append({
            "similarity_group": bin_label,
            "threshold_type": "bin",
            "n_compounds": int(len(subset)),
            "fraction_of_evaluation_set": float(len(subset) / total_predictions) if total_predictions else np.nan,
            "n_positive": int(np.sum(y_true == 1)) if len(y_true) else 0,
            "n_negative": int(np.sum(y_true == 0)) if len(y_true) else 0,
            **metrics,
        })

    threshold_performance = pd.DataFrame(threshold_rows)
    bin_performance = pd.DataFrame(bin_rows)

    predictions_path = os.path.join(output_dir, f"{eval_name}_similarity_predictions.csv")
    threshold_path = os.path.join(output_dir, f"{eval_name}_similarity_threshold_performance.csv")
    bin_path = os.path.join(output_dir, f"{eval_name}_similarity_bin_performance.csv")

    prediction_overlap.to_csv(predictions_path, index=False)
    threshold_performance.to_csv(threshold_path, index=False)
    bin_performance.to_csv(bin_path, index=False)

    print(f"{eval_name} similarity-threshold performance:")
    print(threshold_performance.to_string(index=False))
    print(f"{eval_name} similarity-bin performance:")
    print(bin_performance.to_string(index=False))
    logger.info("%s similarity threshold performance saved to %s", eval_name, threshold_path)
    logger.info("%s similarity bin performance saved to %s", eval_name, bin_path)

    return prediction_overlap, threshold_performance, bin_performance


def write_class_similarity_by_train_test_bin_report(
    prediction_overlap,
    eval_name,
    output_dir=input_dir,
):
    """Measure active-vs-decoy similarity within each existing train/test bin.

    Similarity_Bin is already defined from each test compound's nearest
    training-set neighbor. Inside each of those bins, this report asks a
    different question: how similar are the test-set actives and decoys to one
    another? It reports both all active-decoy pair similarities and each test
    compound's nearest opposite-class neighbor.
    """
    bin_labels = ["< 0.25", "0.25-<0.50", "0.50-<0.75", "0.75-1.00"]
    rows = []
    pair_rows = []
    nearest_rows = []

    for bin_label in bin_labels:
        subset = prediction_overlap[
            prediction_overlap["Similarity_Bin"].astype(str).eq(bin_label)
        ].reset_index(drop=True).copy()
        subset["_fingerprint"] = subset["SMILES"].map(smiles_to_morgan_fp)
        subset = subset[subset["_fingerprint"].notna()].reset_index(drop=True)

        actives = subset[subset["True_Label"].astype(int).eq(1)].reset_index(drop=True)
        decoys = subset[subset["True_Label"].astype(int).eq(0)].reset_index(drop=True)

        pairwise_values = []
        active_to_decoy_nn = []
        decoy_to_active_nn = []

        for _, active_row in actives.iterrows():
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(active_row["_fingerprint"], decoys["_fingerprint"].tolist()),
                dtype=float,
            )
            if len(similarities) == 0:
                continue
            nearest_idx = int(np.argmax(similarities))
            nearest_similarity = float(similarities[nearest_idx])
            active_to_decoy_nn.append(nearest_similarity)
            nearest_decoy = decoys.iloc[nearest_idx]
            nearest_rows.append({
                "similarity_group": bin_label,
                "query_class": "active",
                "query_title": active_row["Title"],
                "query_smiles": active_row["SMILES"],
                "nearest_opposite_class_title": nearest_decoy["Title"],
                "nearest_opposite_class_smiles": nearest_decoy["SMILES"],
                "nearest_opposite_class_tanimoto": nearest_similarity,
            })
            for decoy_idx, similarity in enumerate(similarities):
                decoy_row = decoys.iloc[decoy_idx]
                similarity = float(similarity)
                pairwise_values.append(similarity)
                pair_rows.append({
                    "similarity_group": bin_label,
                    "active_title": active_row["Title"],
                    "decoy_title": decoy_row["Title"],
                    "active_decoy_tanimoto": similarity,
                })

        for _, decoy_row in decoys.iterrows():
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(decoy_row["_fingerprint"], actives["_fingerprint"].tolist()),
                dtype=float,
            )
            if len(similarities) == 0:
                continue
            nearest_idx = int(np.argmax(similarities))
            nearest_similarity = float(similarities[nearest_idx])
            decoy_to_active_nn.append(nearest_similarity)
            nearest_active = actives.iloc[nearest_idx]
            nearest_rows.append({
                "similarity_group": bin_label,
                "query_class": "decoy",
                "query_title": decoy_row["Title"],
                "query_smiles": decoy_row["SMILES"],
                "nearest_opposite_class_title": nearest_active["Title"],
                "nearest_opposite_class_smiles": nearest_active["SMILES"],
                "nearest_opposite_class_tanimoto": nearest_similarity,
            })

        pairwise_values = np.asarray(pairwise_values, dtype=float)
        active_to_decoy_nn = np.asarray(active_to_decoy_nn, dtype=float)
        decoy_to_active_nn = np.asarray(decoy_to_active_nn, dtype=float)
        pooled_nn = np.concatenate([active_to_decoy_nn, decoy_to_active_nn])

        rows.append({
            "similarity_group": bin_label,
            "n_compounds": int(len(subset)),
            "n_active": int(len(actives)),
            "n_decoy": int(len(decoys)),
            "n_active_decoy_pairs": int(len(pairwise_values)),
            "mean_active_decoy_pairwise_tanimoto": float(np.mean(pairwise_values)) if len(pairwise_values) else np.nan,
            "median_active_decoy_pairwise_tanimoto": float(np.median(pairwise_values)) if len(pairwise_values) else np.nan,
            "max_active_decoy_pairwise_tanimoto": float(np.max(pairwise_values)) if len(pairwise_values) else np.nan,
            "mean_active_to_decoy_nearest_tanimoto": float(np.mean(active_to_decoy_nn)) if len(active_to_decoy_nn) else np.nan,
            "median_active_to_decoy_nearest_tanimoto": float(np.median(active_to_decoy_nn)) if len(active_to_decoy_nn) else np.nan,
            "max_active_to_decoy_nearest_tanimoto": float(np.max(active_to_decoy_nn)) if len(active_to_decoy_nn) else np.nan,
            "mean_decoy_to_active_nearest_tanimoto": float(np.mean(decoy_to_active_nn)) if len(decoy_to_active_nn) else np.nan,
            "median_decoy_to_active_nearest_tanimoto": float(np.median(decoy_to_active_nn)) if len(decoy_to_active_nn) else np.nan,
            "max_decoy_to_active_nearest_tanimoto": float(np.max(decoy_to_active_nn)) if len(decoy_to_active_nn) else np.nan,
            "fraction_nearest_opposite_class_ge_0_50": float(np.mean(pooled_nn >= 0.50)) if len(pooled_nn) else np.nan,
        })

    summary_df = pd.DataFrame(rows)
    pairwise_df = pd.DataFrame(pair_rows)
    nearest_df = pd.DataFrame(nearest_rows)

    summary_path = os.path.join(output_dir, f"{eval_name}_class_similarity_by_train_test_bin.csv")
    pairwise_path = os.path.join(output_dir, f"{eval_name}_active_decoy_pairwise_by_train_test_bin.csv")
    nearest_path = os.path.join(output_dir, f"{eval_name}_opposite_class_nearest_neighbors_by_train_test_bin.csv")
    pairwise_plot_path = os.path.join(output_dir, f"{eval_name}_active_decoy_pairwise_by_train_test_bin.png")
    nearest_plot_path = os.path.join(output_dir, f"{eval_name}_opposite_class_nearest_neighbors_by_train_test_bin.png")

    summary_df.to_csv(summary_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    nearest_df.to_csv(nearest_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=False)
    for ax, bin_label in zip(axes.flat, bin_labels):
        values = pairwise_df.loc[
            pairwise_df["similarity_group"].eq(bin_label),
            "active_decoy_tanimoto",
        ]
        ax.hist(values, bins=np.linspace(0, 1, 41), color="#2f6f9f", edgecolor="white", linewidth=0.4)
        ax.set_title(bin_label)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    fig.supxlabel("Active-decoy pairwise Tanimoto similarity")
    fig.supylabel("Number of active-decoy pairs")
    fig.suptitle(f"{eval_name} active-decoy pairwise similarity within train/test bins")
    fig.tight_layout()
    fig.savefig(pairwise_plot_path, dpi=600)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=False)
    for ax, bin_label in zip(axes.flat, bin_labels):
        bin_nearest = nearest_df[nearest_df["similarity_group"].eq(bin_label)]
        for query_class, color in (("active", "#b23a48"), ("decoy", "#2f6f9f")):
            values = bin_nearest.loc[
                bin_nearest["query_class"].eq(query_class),
                "nearest_opposite_class_tanimoto",
            ]
            ax.hist(
                values,
                bins=np.linspace(0, 1, 41),
                alpha=0.6,
                label=query_class,
                color=color,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_title(bin_label)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.supxlabel("Nearest opposite-class Tanimoto similarity")
    fig.supylabel("Number of scaffold-test molecules")
    fig.suptitle(f"{eval_name} nearest opposite-class similarity within train/test bins")
    fig.tight_layout()
    fig.savefig(nearest_plot_path, dpi=600)
    plt.close(fig)

    print(f"{eval_name} active/decoy similarity within train/test bins:")
    print(summary_df.to_string(index=False))
    logger.info("%s class similarity by train/test bin saved to %s", eval_name, summary_path)

    return summary_df, pairwise_df, nearest_df


def report_train_test_tanimoto(train_smiles, test_smiles):
    train_fps = [
        fp for fp in (smiles_to_morgan_fp(smiles) for smiles in train_smiles)
        if fp is not None
    ]
    test_fps = [
        fp for fp in (smiles_to_morgan_fp(smiles) for smiles in test_smiles)
        if fp is not None
    ]

    nearest_train_tanimoto = []
    mean_train_tanimoto = []
    pairwise_similarity_sum = 0.0
    pairwise_similarity_count = 0
    for test_fp in test_fps:
        similarities = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        similarities_array = np.asarray(similarities, dtype=float)
        nearest_train_tanimoto.append(float(similarities_array.max()))
        mean_train_tanimoto.append(float(similarities_array.mean()))
        pairwise_similarity_sum += float(similarities_array.sum())
        pairwise_similarity_count += len(similarities_array)

    nearest_train_tanimoto = np.asarray(nearest_train_tanimoto, dtype=float)
    mean_train_tanimoto = np.asarray(mean_train_tanimoto, dtype=float)
    summary = {
        "n_train_fingerprints": len(train_fps),
        "n_test_fingerprints": len(test_fps),
        "n_train_test_pairs": int(pairwise_similarity_count),
        "mean_pairwise_train_test_tanimoto": float(pairwise_similarity_sum / pairwise_similarity_count),
        "mean_test_average_train_tanimoto": float(mean_train_tanimoto.mean()),
        "median_test_average_train_tanimoto": float(np.median(mean_train_tanimoto)),
        "mean_nearest_train_tanimoto": float(nearest_train_tanimoto.mean()),
        "median_nearest_train_tanimoto": float(np.median(nearest_train_tanimoto)),
        "min_nearest_train_tanimoto": float(nearest_train_tanimoto.min()),
        "max_nearest_train_tanimoto": float(nearest_train_tanimoto.max()),
        "fraction_test_nearest_train_tanimoto_ge_0_7": float(np.mean(nearest_train_tanimoto >= 0.7)),
        "fraction_test_nearest_train_tanimoto_ge_0_8": float(np.mean(nearest_train_tanimoto >= 0.8)),
        "fraction_test_nearest_train_tanimoto_ge_0_9": float(np.mean(nearest_train_tanimoto >= 0.9)),
    }

    logger.info("Train/test Tanimoto summary: %s", summary)
    with open(os.path.join(input_dir, "train_test_tanimoto_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Train/test Tanimoto summary:")
    for name, value in summary.items():
        print(f"  {name}: {value}")
    return summary


class EdgeBlock(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, dropout_rate):
        super(EdgeBlock, self).__init__()
        print(f"Initializing Training with input_dim: {input_dim}, hidden_dim: {hidden_dim}, dropout_rate: {dropout_rate}")        
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim * 2 + edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)
class NodeBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(NodeBlock, self).__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),  
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = x[row]  
        out = torch.cat([edge_attr, out], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0), reduce='max')
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)
class GlobalBlock(nn.Module):
    def __init__(self, hidden_dim, global_dim, dropout_rate):
        super(GlobalBlock, self).__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  
        )
    def forward(self, x, edge_index, edge_attr, u, batch):
        x_pooled = global_max_pool(x, batch)
        num_graphs = x_pooled.size(0)
        u_reshaped = u.view(num_graphs, -1)      
        out = torch.cat([x_pooled, u_reshaped], dim=1)
        out = self.global_mlp(out)
        return out        
class InteractionNetwork(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, global_dim, dropout_rate):
        super(InteractionNetwork, self).__init__()
        self.interactionnetwork = MetaLayer(
            EdgeBlock(input_dim, edge_dim, hidden_dim, dropout_rate),
            NodeBlock(input_dim, hidden_dim, dropout_rate),
            GlobalBlock(hidden_dim, global_dim, dropout_rate)
        )
        self.bn = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(edge_dim)
    def forward(self, x, edge_index, edge_attr, u, batch):
        x = self.bn(x)
        edge_attr = self.bn2(edge_attr)
        x, edge_attr, u = self.interactionnetwork(x, edge_index, edge_attr, u, batch)
        return u
class MolecularDataset:
    def __init__(self, smiles_list, Titles_list, labels=None, normalization_stats=None):
        self.smiles_list = smiles_list.copy()  # Copy to avoid mutating the original input
        self.Titles_list = Titles_list.copy()
        self.labels = labels.copy() if labels is not None else [None] * len(smiles_list)
        self.data_list = []
        self.hybridization_dict = {
            Chem.rdchem.HybridizationType.SP: 0,
            Chem.rdchem.HybridizationType.SP2: 0.5,
            Chem.rdchem.HybridizationType.SP3: 1,
        }
        self.degree_dict = {1: 0, 2: 0.5, 3: 1}
        self.explicit_dict = {1: 0, 2: 0.33, 3: 0.66, 4: 1}
        self.processed_count = 0
        if normalization_stats is None:
            logger.info("Calculating min and max values for normalization.")
            self.mins, self.maxs, self.min_bond_length, self.max_bond_length = self.calculate_min_max(smiles_list)
        else:
            logger.info("Using supplied training-set normalization values.")
            self.mins, self.maxs, self.min_bond_length, self.max_bond_length = normalization_stats
        logger.info("Converting SMILES to data objects.")
        i = 0  
        while i < len(self.smiles_list):  
            smiles = self.smiles_list[i]
            Title = self.Titles_list[i]
            label = self.labels[i]
            try:
                data = self.smiles_to_data(smiles, Title, label)
                if data is not None:
                    self.data_list.append(data)
                    self.processed_count += 1
                else:
                    logger.warning(f"Skipping invalid molecule: Title: {Title}, SMILES: {smiles}")
                    del self.smiles_list[i]
                    del self.Titles_list[i]
                    del self.labels[i]
                    continue
            except Exception as e:
                logger.error(f"Failed to process molecule: Title: {Title}, SMILES: {smiles}, Error: {e}")
                del self.smiles_list[i]
                del self.Titles_list[i]
                del self.labels[i]
                continue 
            i += 1  
        logger.info(f"Processed {self.processed_count} valid molecules out of {len(smiles_list)} provided.")        
    def smiles_to_data(self, smiles, Title, label=None, output_dir="molecule_images"):
        try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Failed to parse SMILES: {smiles}")
                    return None  # Ensure SMILES and corresponding data are removed in calling function.                
                mol = self.correct_atom_types(mol)
                mol_with_h = Chem.AddHs(mol)
                try:
                    AllChem.EmbedMolecule(mol_with_h)
                    AllChem.MMFFOptimizeMolecule(mol_with_h)            
                    conf_with_h = mol_with_h.GetConformer()
                    mol = Chem.RemoveHs(mol_with_h)
                    conf = Chem.Conformer(mol.GetNumAtoms())
                    for atom_id in range(mol.GetNumAtoms()):
                        pos = conf_with_h.GetAtomPosition(atom_id)
                        conf.SetAtomPosition(atom_id, pos)
                    mol.AddConformer(conf)
                    Chem.SanitizeMol(mol)                     
                except Exception as e:
                    logger.error(f"An error occurred during sanitization or optimization: {e}, SMILES: {smiles}")
                    return None                  
                try:
                    atom_features = self.get_atom_features(mol)
                    edge_index, edge_attr = self.get_edge_index_and_features(mol, conf)
                    global_features = self.get_global_features(mol, conf)
                    
                    if label is not None:
                        target = torch.tensor([label], dtype=torch.float).reshape(-1, 1)
                        data = Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr, u=global_features, y=target)
                    else:
                        data = Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr, u=global_features)                    
                    data.smiles = smiles
                    data.Title = Title
                    self.save_molecule_image(mol, Title, output_dir)
                    logger.info(f"Processed {self.processed_count} molecules so far.")                    
                    return data                
                except Exception as e:
                    logger.error(f"Failed to process molecule features: {Title}, SMILES: {smiles}, Error: {e}")
                    return None             
        except Exception as e:
            logger.error(f"General failure processing SMILES: {smiles}, Error: {e}")
            return None         
    def calculate_min_max(self, smiles_list):
        features = {
            'NumHeavyAtoms': [],
            'NumHDonors': [],
            'NumRotatableBonds': [],
            'TPSA': [],
            'NumHacceptors': [],
            'LogP': [],
            'FractionSP3': [],
            'AromaticProportion': [],
            'AtomicNumber': [],
            'Compactness': []
        }        
        min_bond_length = float('inf')
        max_bond_length = float('-inf')
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
                continue
            mol = self.correct_atom_types(mol)
            mol_with_h = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(mol_with_h)
                AllChem.MMFFOptimizeMolecule(mol_with_h)                
                conf_with_h = mol_with_h.GetConformer()
                mol = Chem.RemoveHs(mol_with_h)
                conf = Chem.Conformer(mol.GetNumAtoms())
                for atom_id in range(mol.GetNumAtoms()):
                    pos = conf_with_h.GetAtomPosition(atom_id)
                    conf.SetAtomPosition(atom_id, pos)
                mol.AddConformer(conf)
                Chem.SanitizeMol(mol)            
            except Exception as e:
                logger.error(f"An error occurred during sanitization: {e} , {smiles}")
                continue
            heavy_atom_count = Descriptors.HeavyAtomCount(mol)
            num_h_donors = Descriptors.NumHDonors(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            tpsa = Descriptors.TPSA(mol)
            num_h_acceptors = Descriptors.NumHAcceptors(mol)
            logp = Descriptors.MolLogP(mol)
            fraction_sp3 = Descriptors.FractionCSP3(mol)
            aromatic_proportion = sum(atom.GetIsAromatic() for atom in mol.GetAtoms()) / heavy_atom_count
            compactness = self.calculate_radius_of_gyration(mol, conf)
            features['NumHeavyAtoms'].append(heavy_atom_count)
            features['NumHDonors'].append(num_h_donors)
            features['NumRotatableBonds'].append(rotatable_bonds)
            features['TPSA'].append(tpsa)
            features['NumHacceptors'].append(num_h_acceptors)
            features['LogP'].append(logp)
            features['FractionSP3'].append(fraction_sp3)
            features['AromaticProportion'].append(aromatic_proportion)
            features['Compactness'].append(compactness)
            for atom in mol.GetAtoms():
                features['AtomicNumber'].append(atom.GetAtomicNum())           
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_length = Chem.rdMolTransforms.GetBondLength(conf, i, j)
                min_bond_length = min(min_bond_length, bond_length)
                max_bond_length = max(max_bond_length, bond_length)
        if all(len(values) > 0 for values in features.values()):
            mins = {key: np.min(values) for key, values in features.items()}
            maxs = {key: np.max(values) for key, values in features.items()}
            return mins, maxs, min_bond_length, max_bond_length
        else:
            raise ValueError("No valid molecules were processed to calculate min and max values.")
    def min_max_normalize(self, value, min_value, max_value):
        if max_value == min_value:
            return 0
        return (value - min_value) / (max_value - min_value)
    def calculate_radius_of_gyration(self, mol, conf):
        coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        masses = np.array([atom.GetMass() for atom in mol.GetAtoms()])
        total_mass = np.sum(masses)
        center_of_mass = np.sum(coords.T * masses, axis=1) / total_mass
        rg_square = np.sum(masses * np.sum((coords - center_of_mass) ** 2, axis=1)) / total_mass
        radius_of_gyration = np.sqrt(rg_square)
        return radius_of_gyration
    def get_global_features(self, mol, conf):
        global_features = [
            self.min_max_normalize(Descriptors.HeavyAtomCount(mol), self.mins['NumHeavyAtoms'], self.maxs['NumHeavyAtoms']),
            Descriptors.RingCount(mol) / 6,
            len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)) / 5,
            self.min_max_normalize(Descriptors.NumHDonors(mol), self.mins['NumHDonors'], self.maxs['NumHDonors']),
            self.min_max_normalize(Descriptors.NumRotatableBonds(mol), self.mins['NumRotatableBonds'], self.maxs['NumRotatableBonds']),
            self.min_max_normalize(Descriptors.TPSA(mol), self.mins['TPSA'], self.maxs['TPSA']),
            self.min_max_normalize(Descriptors.MolLogP(mol), self.mins['LogP'], self.maxs['LogP']),
            self.min_max_normalize(Descriptors.FractionCSP3(mol), self.mins['FractionSP3'], self.maxs['FractionSP3']),
            self.min_max_normalize(self.calculate_radius_of_gyration(mol, conf), self.mins['Compactness'], self.maxs['Compactness']),
        ]
        return torch.tensor(global_features, dtype=torch.float)
    def correct_atom_types(self, mol):
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "Se" and atom.GetFormalCharge() == 2:
                print(f"Correcting atom type for: {atom.GetSymbol()}, charge: {atom.GetFormalCharge()}")
                atom.SetFormalCharge(2)
        return mol   
    def save_molecule_image(self, mol, Title, output_dir="molecule_images", img_size=(300, 300)):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img = Draw.MolToImage(mol, size=img_size)
        img_path = os.path.join(output_dir, f"molecule_{Title}.png")
        img.save(img_path)
    def degree_to_index(self, degree):
        return self.degree_dict.get(degree, 0)
    def explicit_to_index(self, explicit):
        return self.explicit_dict.get(explicit, 0)          
    def hybridization_to_index(self, hybridization):
        return self.hybridization_dict.get(hybridization, 0)
    def get_atom_features(self, mol):
        atom_features = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1: 
                continue
            atom_feature = [
                self.min_max_normalize(atom.GetAtomicNum(), self.mins['AtomicNumber'], self.maxs['AtomicNumber']),
                self.degree_to_index(atom.GetDegree()),
                atom.GetIsAromatic(),
                self.explicit_to_index(atom.GetExplicitValence()),
                self.hybridization_to_index(atom.GetHybridization()),
            ]
            atom_features.append(atom_feature)
        return torch.tensor(atom_features, dtype=torch.float)
    def get_ring_size_feature(self, bond):
        if not bond.IsInRing():
            return 0.0
        elif bond.IsInRingSize(4):
            return 0.16
        elif bond.IsInRingSize(5):
            return 0.33
        elif bond.IsInRingSize(6):
            return 0.5
        elif bond.IsInRingSize(7):
            return 0.66
        elif bond.IsInRingSize(8):
            return 0.82
        else:
            return 1.0  # Default value for rings of other sizes or no ring           
    def get_edge_index_and_features(self, mol, conf):
        edge_index = []
        edge_attr = []
        try:
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_length = Chem.rdMolTransforms.GetBondLength(conf, i, j)
                ring_size_feature = self.get_ring_size_feature(bond)

                normalized_bond_length = (bond_length - self.min_bond_length) / (self.max_bond_length - self.min_bond_length)                
                edge_feature = [
                    normalized_bond_length,
                    bond.GetBondTypeAsDouble() / 2,
                    (1 if bond.GetIsConjugated() else 0),
                    ring_size_feature
                ]               
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append(edge_feature)
                edge_attr.append(edge_feature)  
        except Exception as e:
            print(f"Error processing bond features for molecule: {e}")
            return None, None   
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous(), torch.tensor(edge_attr, dtype=torch.float)
    def len(self):
        return len(self.data_list)
    def get(self, idx):
        return self.data_list[idx]
    def indices(self):
        return range(self.len())
    def __len__(self):
        return self.len()
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.get(idx)
        else:
            return self.__class__(self.smiles_list[idx], self.Titles_list[idx], self.labels[idx])
class GNNTrainer:
    def __init__(self, smiles_list, labels=None, Titles_list=None, hidden_channels=64, num_node_features=128, global_dim=10, lr=0.001, weight_decay=0.0, edge_dim=5, batch_size=32, k_folds=5, dropout_rate=0.5):
        self.smiles_list = smiles_list
        self.labels = labels
        self.Titles_list = Titles_list
        self.hidden_channels = hidden_channels
        self.num_node_features = num_node_features  
        self.global_dim = global_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.k_folds = k_folds
        self.epochs = epochs  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout_rate = dropout_rate
        self.edge_dim = edge_dim
        self.threshold = threshold
    def setup_model(self):
        self.model = InteractionNetwork(self.num_node_features, self.edge_dim, self.hidden_channels, self.global_dim, self.dropout_rate).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.criterion = nn.BCEWithLogitsLoss()
    def train(self, train_loader):
        self.model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.edge_attr, data.u, data.batch)
            loss = self.criterion(out, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data.num_graphs
        return total_loss / len(train_loader.dataset)
    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total_loss = 0
        all_targets = []
        all_preds = []
        compound_Titles = []  
        for data in loader: 
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.edge_attr, data.u, data.batch)
            loss = self.criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            pred = (out > self.threshold).float()
            correct += (pred == data.y).sum().item()
            all_targets.extend(data.y.cpu().numpy().flatten())
            all_preds.extend(out.detach().cpu().numpy().flatten())
            compound_Titles.extend([d.Title for d in data.to_data_list()])  
        all_preds_binarized = [1 if p > self.threshold else 0 for p in all_preds]
        accuracy = accuracy_score(all_targets, all_preds_binarized)
        precision = precision_score(all_targets, all_preds_binarized)
        recall = recall_score(all_targets, all_preds_binarized)
        f1 = f1_score(all_targets, all_preds_binarized)
        return accuracy, precision, recall, f1, total_loss / len(loader.dataset), all_targets, all_preds, compound_Titles, all_preds_binarized
    def cross_validate(self, num_epochs=epochs):
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=52)
        fold_train_accuracies = []
        fold_val_accuracies = []
        fold_val_losses = []
        fold_train_losses = []
        fold_train_f1s = []
        fold_val_f1s = []
        fold_epoch_times = []
        all_targets = []
        all_preds = []
        fpr_list = []
        tpr_list = []
        auc_list = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.smiles_list)):
            print(f'Fold {fold + 1}/{self.k_folds}')
            smiles_train = [self.smiles_list[i] for i in train_idx]
            Titles_train = [self.Titles_list[i] for i in train_idx]
            labels_train = [self.labels[i] for i in train_idx]
            labels_count = np.bincount(labels_train)
            weights = 1. / labels_count
            samples_weights = weights[labels_train]
            sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)           
            smiles_val = [self.smiles_list[i] for i in val_idx]
            labels_val = [self.labels[i] for i in val_idx]
            Titles_val = [self.Titles_list[i] for i in val_idx]
            train_dataset = MolecularDataset(smiles_train, Titles_train, labels_train)
            val_dataset = MolecularDataset(smiles_val, Titles_val, labels_val)
            train_loader = GeometricDataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
            val_loader = GeometricDataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            self.setup_model()
            epoch_train_losses = []
            epoch_val_losses = []
            epoch_train_accuracies = []
            epoch_val_accuracies = []
            epoch_train_f1s = []
            epoch_val_f1s = []
            epoch_times = []
            for epoch in range(num_epochs):
                start_time = time.time()
                train_loss = self.train(train_loader)
                train_acc, train_prec, train_rec, train_f1, train_loss, _, _, _, _ = self.evaluate(train_loader)
                val_acc, val_prec, val_rec, val_f1, val_loss, targets, preds, _, _ = self.evaluate(val_loader)
                epoch_time = time.time() - start_time
                epoch_train_losses.append(train_loss)
                epoch_val_losses.append(val_loss)
                epoch_train_accuracies.append(train_acc)
                epoch_val_accuracies.append(val_acc)
                epoch_train_f1s.append(train_f1)
                epoch_val_f1s.append(val_f1)
                epoch_times.append(epoch_time)
                print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                      f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, Time: {epoch_time:.2f}s')
                fpr, tpr, _ = roc_curve(targets, preds)
                roc_auc = auc(fpr, tpr)
                fpr_list.append(fpr)
                tpr_list.append(tpr)
                auc_list.append(roc_auc)
            fold_train_accuracies.append(epoch_train_accuracies)
            fold_val_accuracies.append(epoch_val_accuracies)
            fold_val_losses.append(epoch_val_losses)
            fold_train_losses.append(epoch_train_losses)
            fold_train_f1s.append(epoch_train_f1s)
            fold_val_f1s.append(epoch_val_f1s)
            fold_epoch_times.append(epoch_times)
            all_targets.extend(targets)
            all_preds.extend(preds)
        avg_train_acc = torch.tensor(fold_train_accuracies).mean(dim=0).tolist()
        avg_val_acc = torch.tensor(fold_val_accuracies).mean(dim=0).tolist()
        avg_val_loss = torch.tensor(fold_val_losses).mean(dim=0).tolist()
        avg_train_loss = torch.tensor(fold_train_losses).mean(dim=0).tolist()
        avg_train_f1 = torch.tensor(fold_train_f1s).mean(dim=0).tolist()
        avg_val_f1 = torch.tensor(fold_val_f1s).mean(dim=0).tolist()
        avg_epoch_time = torch.tensor(fold_epoch_times).mean(dim=0).tolist()
        print(f'Average Train Accuracy: {avg_train_acc[-1]:.4f}')
        print(f'Average Validation Accuracy: {avg_val_acc[-1]:.4f}')
        print(f'Average Train Loss: {avg_train_loss[-1]:.4f}')
        print(f'Average Validation Loss: {avg_val_loss[-1]:.4f}')
        print(f'Average Train F1 Score: {avg_train_f1[-1]:.4f}')
        print(f'Average Validation F1 Score: {avg_val_f1[-1]:.4f}')
        print(f'Average Epoch Time: {avg_epoch_time[-1]:.2f}s')
        self.plot_metrics(avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc, avg_train_f1, avg_val_f1, avg_epoch_time)
        self.plot_roc_curve(fpr_list, tpr_list, auc_list)
    def predict(self, smiles_list, compound_Titles, output_csv=None, labels=None, normalization_stats=None):
        self.model.eval()
        prediction_scores = []
        prediction_probabilities = []
        predicted_labels = []
        test_dataset = MolecularDataset(
            smiles_list,
            compound_Titles,
            labels=labels,
            normalization_stats=normalization_stats,
        )
        test_loader = GeometricDataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)    
        smiles_strings = []    
        predicted_Titles = [] 
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Predicting"):
                if data is None:
                    print("Warning: Found None in data loader.")
                    continue
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.edge_attr, data.u, data.batch)
                pred = (out > self.threshold).float()
                prediction_scores.extend(out.detach().cpu().numpy().flatten())
                prediction_probabilities.extend(torch.sigmoid(out).detach().cpu().numpy().flatten())
                predicted_labels.extend(pred.detach().cpu().numpy().astype(int).flatten())
                smiles_strings.extend([d.smiles for d in data.to_data_list()])
                predicted_Titles.extend([d.Title for d in data.to_data_list()])
        results = pd.DataFrame({
            "SMILES": smiles_strings,
            "Compound Title": predicted_Titles,
            "Prediction Score": prediction_scores,
            "Prediction Probability": prediction_probabilities,
            "Predicted Label": predicted_labels,
        })
        if labels is not None:
            results.insert(2, "True Label", test_dataset.labels)
        if output_csv:
            results.to_csv(output_csv, index=False)
        return results
    def plot_metrics(self, train_losses, val_losses, train_accuracies, val_accuracies, train_f1s, val_f1s, epoch_times):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(18, 12))
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, label='Training loss')
        plt.plot(epochs, val_losses, label='Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.subplot(2, 2, 2)
        plt.plot(epochs, train_accuracies, label='Training accuracy')
        plt.plot(epochs, val_accuracies, label='Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.subplot(2, 2, 3)
        plt.plot(epochs, train_f1s, label='Training F1 Score')
        plt.plot(epochs, val_f1s, label='Validation F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title('Training and Validation F1 Score')
        plt.subplot(2, 2, 4)
        plt.plot(epochs, epoch_times, label='Epoch Time')
        plt.xlabel('Epochs')
        plt.ylabel('Time (s)')
        plt.legend()
        plt.title('Epoch Processing Time')
        plt.tight_layout()
        plt.savefig("3_metrics.png")
        plt.show()
    def plot_roc_curve(self, fpr_list, tpr_list, auc_list):
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        for i in range(len(fpr_list)):
            interp_tpr = np.interp(mean_fpr, fpr_list[i], tpr_list[i])
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc_list[i])        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)        
        plt.figure()
        plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('validation_roc_plot.png')
        #plt.show()
report_hyperparameters()

dataset = pd.read_csv('training_dataset.csv')
smiles = dataset['SMILES'].tolist()
labels = dataset['Label'].astype(int).tolist()
Titles = dataset['Title'].tolist()


(
    train_smiles,
    train_labels,
    train_Titles,
    smiles_test,
    labels_test,
    Titles_test,
    train_idx,
    test_idx,
) = scaffold_split(
    smiles,
    labels,
    Titles,
    test_fraction=scaffold_test_fraction,
    return_indices=True,
)

train_metadata = dataset.iloc[train_idx].reset_index(drop=True)
scaffold_test_metadata = dataset.iloc[test_idx].reset_index(drop=True)

report_train_test_tanimoto(train_smiles, smiles_test)
scaffold_overlap_report, _, _ = write_train_eval_tanimoto_overlap_report(
    train_metadata,
    scaffold_test_metadata,
    eval_name="scaffold_test",
    output_dir=input_dir,
)

train_dataset = MolecularDataset(train_smiles, train_Titles, train_labels)
print("Train dataset size:", len(train_dataset))
normalization_stats = (
    train_dataset.mins,
    train_dataset.maxs,
    train_dataset.min_bond_length,
    train_dataset.max_bond_length,
)
successful_labels_train = train_dataset.labels
if len(successful_labels_train) == 0:
    raise ValueError("`successful_labels_train` is empty. Check the dataset preparation.")
labels_count = np.bincount(successful_labels_train)
weights = 1. / labels_count
samples_weights = weights[successful_labels_train]
sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)
train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

test_dataset = MolecularDataset(
    smiles_test,
    Titles_test,
    labels_test,
    normalization_stats=normalization_stats,
)
print("Scaffold test dataset size:", len(test_dataset))
test_loader = GeometricDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

trainer = GNNTrainer(
    smiles_list=train_smiles,
    labels=train_labels,
    Titles_list=train_Titles,
    hidden_channels=hidden_channels,
    num_node_features=num_node_features,
    global_dim=global_dim,
    lr=lr,
    weight_decay=weight_decay,
    edge_dim=edge_dim,
    batch_size=batch_size,
    k_folds=k_folds,
    dropout_rate=dropout_rate,
)
def train_model(self, train_loader, num_epochs):
    for epoch in range(num_epochs):
        epoch_loss = self.train(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
setattr(GNNTrainer, "train_model", train_model)
trainer.setup_model()
trainer.train_model(train_loader, epochs)


_, _, _, _, _, all_targets, all_preds, compound_Titles, all_preds_binarized = trainer.evaluate(test_loader)
f1 = f1_score(all_targets, all_preds_binarized)
accuracy = accuracy_score(all_targets, all_preds_binarized)
print(f'F1 Score: {f1:.4f}')
print(f'Accuracy: {accuracy:.4f}')
plot_path = os.path.join(input_dir, "conf_matrix.png")
cm = confusion_matrix(all_targets, all_preds_binarized)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for Scaffold Test Set')
plt.savefig(plot_path)
fpr, tpr, thresholds = roc_curve(all_targets, all_preds)
roc_auc = auc(fpr, tpr)
plot_path = os.path.join(input_dir, "roc_auc_scaffold_test_set.png")
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Scaffold Test Set')
plt.legend(loc="lower right")
plt.savefig(plot_path)
print(f'AUC-ROC: {roc_auc:.4f}')
scaffold_similarity_predictions, _, _ = write_similarity_performance_report(
    scaffold_overlap_report,
    compound_Titles,
    all_targets,
    all_preds,
    all_preds_binarized,
    eval_name="scaffold_test",
    thresholds=(0.25, 0.50, 0.75, 0.90),
    output_dir=input_dir,
)
write_class_similarity_by_train_test_bin_report(
    scaffold_similarity_predictions,
    eval_name="scaffold_test",
    output_dir=input_dir,
)
for Title, true_label, pred_label in zip(compound_Titles, all_targets, all_preds_binarized):
    logger.info(f'Compound: {Title}, True Label: {true_label}, Predicted Label: {pred_label}')
