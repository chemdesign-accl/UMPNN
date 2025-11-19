#!/usr/bin/env python
# coding: utf-8
import logging
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
from rdkit.Chem import (
    AllChem, Draw, Descriptors)
import csv
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
edge_dim = 4
global_dim = 9
num_node_features = 5
# Set up logging
log_file_path = os.path.join(input_dir, "logfile.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path, 
    filemode='w' 
)
logger = logging.getLogger(__name__)
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
    def __init__(self, smiles_list, Titles_list, labels=None):
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
        logger.info("Calculating min and max values for normalization.")
        self.mins, self.maxs, self.min_bond_length, self.max_bond_length = self.calculate_min_max(smiles_list)
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
    def __init__(self, smiles_list, labels=None, Titles_list=None, hidden_channels=64, num_node_features=128, global_dim=10, lr=0.001, edge_dim=5, batch_size=32, k_folds=5, dropout_rate=0.5):
        self.smiles_list = smiles_list
        self.labels = labels
        self.Titles_list = Titles_list
        self.hidden_channels = hidden_channels
        self.num_node_features = num_node_features  
        self.global_dim = global_dim
        self.lr = lr
        self.batch_size = batch_size
        self.k_folds = k_folds
        self.epochs = epochs  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout_rate = dropout_rate
        self.edge_dim = edge_dim
        self.threshold = threshold
    def setup_model(self):
        self.model = InteractionNetwork(self.num_node_features, self.edge_dim, self.hidden_channels, self.global_dim, self.dropout_rate).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
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
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
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
    def predict(self, smiles_list, compound_Titles, output_csv=None):
        self.model.eval()
        predictions = []
        test_dataset = MolecularDataset(smiles_list, compound_Titles, labels=None)
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
                predictions.extend(pred.detach().cpu().numpy().flatten())  
                smiles_strings.extend([d.smiles for d in data.to_data_list()])
                predicted_Titles.extend([d.Title for d in data.to_data_list()])
        results = list(zip(smiles_strings, predicted_Titles, predictions))  
        if output_csv:
            with open(output_csv, mode='wt', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['SMILES', 'Compound Title', 'Predicted Label'])  
                writer.writerows(results)  
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
dataset = pd.read_csv('./combined_decoy_EC50_cleaned.csv')
smiles = dataset['SMILES'].tolist()
labels = dataset['Actividad'].tolist()
Titles = dataset['Title'].tolist()
train_dataset = MolecularDataset(smiles, Titles, labels)
print("Train dataset size:", len(train_dataset))
successful_labels_train = labels
if len(successful_labels_train) == 0:
    raise ValueError("`successful_labels_train` is empty. Check the dataset preparation.")
labels_count = np.bincount(successful_labels_train)
weights = 1. / labels_count
samples_weights = weights[successful_labels_train]
sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)
train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
blind_dataset = pd.read_csv('blind_compounds.csv')
blind_smiles = blind_dataset['SMILES'].tolist()
blind_labels = blind_dataset['Actividad'].tolist()
blind_Titles = blind_dataset['Title'].tolist()
blind_dataset = MolecularDataset(blind_smiles, blind_Titles, blind_labels)
print("Test dataset size:", len(blind_dataset))
blind_loader = GeometricDataLoader(blind_dataset, batch_size=batch_size, shuffle=False)
trainer = GNNTrainer(
    smiles_list=smiles,
    labels=labels,
    Titles_list=Titles,
    hidden_channels=hidden_channels,
    num_node_features=num_node_features,
    global_dim=global_dim,
    lr=lr,
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
_, _, _, _, _, all_targets, all_preds, compound_Titles, all_preds_binarized = trainer.evaluate(blind_loader)
f1 = f1_score(all_targets, all_preds_binarized)
accuracy = accuracy_score(all_targets, all_preds_binarized)
print(f'F1 Score: {f1:.4f}')
print(f'Accuracy: {accuracy:.4f}')
plot_path = os.path.join(input_dir, "conf_matrix.png")
cm = confusion_matrix(all_targets, all_preds_binarized)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for Blind Test Set')
plt.savefig(plot_path)
fpr, tpr, thresholds = roc_curve(all_targets, all_preds)
roc_auc = auc(fpr, tpr)
plot_path = os.path.join(input_dir, "roc_auc_blind_set.png")
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Blind Test Set')
plt.legend(loc="lower right")
plt.savefig(plot_path)
print(f'AUC-ROC: {roc_auc:.4f}')
# Print compound Titles along with true and predicted labels
for Title, true_label, pred_label in zip(compound_Titles, all_targets, all_preds_binarized):
    logger.info(f'Compound: {Title}, True Label: {true_label}, Predicted Label: {pred_label}')
