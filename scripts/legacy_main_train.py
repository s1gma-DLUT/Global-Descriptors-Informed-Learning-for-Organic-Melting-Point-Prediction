import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem, RDLogger
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GlobalAttention, global_max_pool
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')


@dataclass
class TrainConfig:
    data_dir: str = 'data'
    outputs_root: str = 'outputs'
    model_name: str = '/home/liutao/pxf/MoLFormer-c3-1.1B'
    split_dir: str = 'splits/scaffold'
    seed: int = 114514
    n_folds: int = 5
    batch_size: int = 256
    num_workers: int = 80
    grad_accum_steps: int = 1
    max_length: int = 200
    max_epochs: int = 300
    freeze_bert_epochs: int = 3
    final_tune_epochs: int = 5
    bert_unfreeze_layers: int = 4
    bert_encoder_lr: float = 1e-6
    bert_projection_lr: float = 1e-5
    dmpnn_lr: float = 3e-4
    fusion_lr: float = 8.19e-5
    bert_encoder_weight_decay: float = 1e-2
    bert_projection_weight_decay: float = 1e-2
    dmpnn_weight_decay: float = 1e-5
    fusion_weight_decay: float = 2.49e-2
    dropout: float = 0.325
    xtb_hidden_dim: int = 128
    common_hidden_dim: int = 192
    xtb_depth: int = 3
    dmpnn_hidden_dim: int = 128
    dmpnn_output_dim: int = 128
    dmpnn_layers: int = 4
    dmpnn_dropout: float = 0.2
    use_rdkit_in_xtb: bool = True
    huber_delta: float = 0.5
    final_lr_scale: float = 0.3
    clip_grad_norm: float = 1.0
    cache_graphs: bool = True
    max_folds_to_run: int = 5
    readout_dim: int = 16
    dynamic_weight_scale: float = 1.0
    bias_hidden_dim: int = 64
    main_branch_noise_std: float = 0.002
    xtb_wd_mult: float = 1.3
    stage2_batch_size: int = 512
    stage2_start_epoch: int = 150
    freeze_main_head_epoch: int = 150
    reg_boost_epoch: int = 250
    reg_boost_factor: float = 1.5
    device: str = ''
    folds_to_run_list: str = ''
    output_tag: str = ''


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description='Frozen-split single-GPU residual-boosting MoLFormer + D-MPNN + XTB/RDKit training.'
    )
    parser.add_argument('--data_dir', type=str, default=TrainConfig.data_dir)
    parser.add_argument('--outputs_root', type=str, default=TrainConfig.outputs_root)
    parser.add_argument('--model_name', type=str, default=TrainConfig.model_name)
    parser.add_argument('--split_dir', type=str, default=TrainConfig.split_dir)
    parser.add_argument('--seed', type=int, default=TrainConfig.seed)
    parser.add_argument('--n_folds', type=int, default=TrainConfig.n_folds)
    parser.add_argument('--batch_size', type=int, default=TrainConfig.batch_size)
    parser.add_argument('--num_workers', type=int, default=TrainConfig.num_workers)
    parser.add_argument('--grad_accum_steps', type=int, default=TrainConfig.grad_accum_steps)
    parser.add_argument('--max_length', type=int, default=TrainConfig.max_length)
    parser.add_argument('--max_epochs', type=int, default=TrainConfig.max_epochs)
    parser.add_argument('--freeze_bert_epochs', type=int, default=TrainConfig.freeze_bert_epochs)
    parser.add_argument('--final_tune_epochs', type=int, default=TrainConfig.final_tune_epochs)
    parser.add_argument('--bert_unfreeze_layers', type=int, default=TrainConfig.bert_unfreeze_layers)
    parser.add_argument('--bert_encoder_lr', type=float, default=TrainConfig.bert_encoder_lr)
    parser.add_argument('--bert_projection_lr', type=float, default=TrainConfig.bert_projection_lr)
    parser.add_argument('--dmpnn_lr', type=float, default=TrainConfig.dmpnn_lr)
    parser.add_argument('--fusion_lr', type=float, default=TrainConfig.fusion_lr)
    parser.add_argument('--bert_encoder_weight_decay', type=float, default=TrainConfig.bert_encoder_weight_decay)
    parser.add_argument('--bert_projection_weight_decay', type=float, default=TrainConfig.bert_projection_weight_decay)
    parser.add_argument('--dmpnn_weight_decay', type=float, default=TrainConfig.dmpnn_weight_decay)
    parser.add_argument('--fusion_weight_decay', type=float, default=TrainConfig.fusion_weight_decay)
    parser.add_argument('--dropout', type=float, default=TrainConfig.dropout)
    parser.add_argument('--xtb_hidden_dim', type=int, default=TrainConfig.xtb_hidden_dim)
    parser.add_argument('--common_hidden_dim', type=int, default=TrainConfig.common_hidden_dim)
    parser.add_argument('--dmpnn_hidden_dim', type=int, default=TrainConfig.dmpnn_hidden_dim)
    parser.add_argument('--dmpnn_output_dim', type=int, default=TrainConfig.dmpnn_output_dim)
    parser.add_argument('--dmpnn_layers', type=int, default=TrainConfig.dmpnn_layers)
    parser.add_argument('--dmpnn_dropout', type=float, default=TrainConfig.dmpnn_dropout)
    parser.add_argument('--huber_delta', type=float, default=TrainConfig.huber_delta)
    parser.add_argument('--final_lr_scale', type=float, default=TrainConfig.final_lr_scale)
    parser.add_argument('--no_cache_graphs', action='store_true')
    parser.add_argument('--max_folds_to_run', type=int, default=TrainConfig.max_folds_to_run)
    parser.add_argument('--readout_dim', type=int, default=TrainConfig.readout_dim)
    parser.add_argument('--dynamic_weight_scale', type=float, default=TrainConfig.dynamic_weight_scale)
    parser.add_argument('--bias_hidden_dim', type=int, default=TrainConfig.bias_hidden_dim)
    parser.add_argument('--main_branch_noise_std', type=float, default=TrainConfig.main_branch_noise_std)
    parser.add_argument('--xtb_wd_mult', type=float, default=TrainConfig.xtb_wd_mult)
    parser.add_argument('--stage2_batch_size', type=int, default=TrainConfig.stage2_batch_size)
    parser.add_argument('--stage2_start_epoch', type=int, default=TrainConfig.stage2_start_epoch)
    parser.add_argument('--freeze_main_head_epoch', type=int, default=TrainConfig.freeze_main_head_epoch)
    parser.add_argument('--reg_boost_epoch', type=int, default=TrainConfig.reg_boost_epoch)
    parser.add_argument('--reg_boost_factor', type=float, default=TrainConfig.reg_boost_factor)
    parser.add_argument('--device', type=str, default=TrainConfig.device)
    parser.add_argument('--folds_to_run_list', type=str, default=TrainConfig.folds_to_run_list)
    parser.add_argument('--output_tag', type=str, default=TrainConfig.output_tag)
    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        outputs_root=args.outputs_root,
        model_name=args.model_name,
        split_dir=args.split_dir,
        seed=args.seed,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        grad_accum_steps=max(1, args.grad_accum_steps),
        max_length=args.max_length,
        max_epochs=args.max_epochs,
        freeze_bert_epochs=args.freeze_bert_epochs,
        final_tune_epochs=args.final_tune_epochs,
        bert_unfreeze_layers=args.bert_unfreeze_layers,
        bert_encoder_lr=args.bert_encoder_lr,
        bert_projection_lr=args.bert_projection_lr,
        dmpnn_lr=args.dmpnn_lr,
        fusion_lr=args.fusion_lr,
        bert_encoder_weight_decay=args.bert_encoder_weight_decay,
        bert_projection_weight_decay=args.bert_projection_weight_decay,
        dmpnn_weight_decay=args.dmpnn_weight_decay,
        fusion_weight_decay=args.fusion_weight_decay,
        dropout=args.dropout,
        xtb_hidden_dim=args.xtb_hidden_dim,
        common_hidden_dim=args.common_hidden_dim,
        dmpnn_hidden_dim=args.dmpnn_hidden_dim,
        dmpnn_output_dim=args.dmpnn_output_dim,
        dmpnn_layers=args.dmpnn_layers,
        dmpnn_dropout=args.dmpnn_dropout,
        use_rdkit_in_xtb=True,
        huber_delta=args.huber_delta,
        final_lr_scale=args.final_lr_scale,
        cache_graphs=not args.no_cache_graphs,
        max_folds_to_run=max(1, min(args.max_folds_to_run, args.n_folds)),
        readout_dim=args.readout_dim,
        dynamic_weight_scale=args.dynamic_weight_scale,
        bias_hidden_dim=args.bias_hidden_dim,
        main_branch_noise_std=args.main_branch_noise_std,
        xtb_wd_mult=args.xtb_wd_mult,
        stage2_batch_size=args.stage2_batch_size,
        stage2_start_epoch=args.stage2_start_epoch,
        freeze_main_head_epoch=args.freeze_main_head_epoch,
        reg_boost_epoch=args.reg_boost_epoch,
        reg_boost_factor=args.reg_boost_factor,
        device=args.device,
        folds_to_run_list=args.folds_to_run_list,
        output_tag=args.output_tag,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_selected_folds(cfg: TrainConfig) -> List[int]:
    if cfg.folds_to_run_list.strip():
        parsed = [int(x.strip()) for x in cfg.folds_to_run_list.split(',') if x.strip()]
        return [fold for fold in parsed if 1 <= fold <= cfg.n_folds]
    return list(range(1, min(cfg.max_folds_to_run, cfg.n_folds) + 1))


def infer_device(cfg: TrainConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


def save_dataframe(df: pd.DataFrame, path_stem: str, index: bool = False) -> str:
    parquet_path = f'{path_stem}.parquet'
    try:
        df.to_parquet(parquet_path, index=index)
        return parquet_path
    except Exception:
        csv_path = f'{path_stem}.csv'
        df.to_csv(csv_path, index=index)
        return csv_path


def append_run_summary(record: Dict[str, Any], outputs_root: str) -> str:
    path = os.path.join(outputs_root, 'runs_summary.csv')
    df = pd.DataFrame([record])
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return path


def get_git_commit(repo_dir: str) -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


# =========================================================
# Frozen split utilities
# =========================================================
def load_frozen_split_manifest(split_dir: str) -> pd.DataFrame:
    manifest_file = os.path.join(split_dir, 'split_manifest.csv')
    if not os.path.exists(manifest_file):
        raise FileNotFoundError(f'Frozen split manifest not found: {manifest_file}')
    manifest_df = pd.read_csv(manifest_file)
    required_cols = {'sample_id', 'smiles', 'assigned_val_fold', 'is_none_scaffold'}
    missing = required_cols - set(manifest_df.columns)
    if missing:
        raise ValueError(f'Frozen split manifest missing columns: {sorted(missing)}')
    return manifest_df


def validate_frozen_manifest_alignment(smiles_list: Sequence[str], manifest_df: pd.DataFrame) -> None:
    if len(manifest_df) != len(smiles_list):
        raise ValueError(
            f'Frozen split manifest length {len(manifest_df)} does not match aligned sample count {len(smiles_list)}'
        )
    manifest_sorted = manifest_df.sort_values('sample_id').reset_index(drop=True)
    expected_ids = list(range(len(smiles_list)))
    actual_ids = manifest_sorted['sample_id'].tolist()
    if actual_ids != expected_ids:
        raise ValueError('Frozen split manifest sample_id is not a complete 0..N-1 sequence.')
    manifest_smiles = manifest_sorted['smiles'].tolist()
    if manifest_smiles != list(smiles_list):
        mismatch_idx = next((i for i, (a, b) in enumerate(zip(manifest_smiles, smiles_list)) if a != b), None)
        raise ValueError(
            f'Frozen split manifest smiles do not align with the current aligned dataset order. '
            f'First mismatch at sample_id={mismatch_idx}: manifest={manifest_smiles[mismatch_idx]!r}, '
            f'data={smiles_list[mismatch_idx]!r}'
        )


def load_frozen_fold_indices(split_dir: str, fold: int, total_samples: int) -> Tuple[List[int], List[int]]:
    train_file = os.path.join(split_dir, f'fold{fold}_train.csv')
    val_file = os.path.join(split_dir, f'fold{fold}_val.csv')

    if not os.path.exists(train_file):
        raise FileNotFoundError(f'Frozen split file not found: {train_file}')
    if not os.path.exists(val_file):
        raise FileNotFoundError(f'Frozen split file not found: {val_file}')

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    required_cols = {'sample_id', 'smiles', 'assigned_val_fold', 'is_none_scaffold'}
    for name, df in [('train', train_df), ('val', val_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f'{name} split file for fold {fold} missing columns: {sorted(missing)}')

    train_idx = train_df['sample_id'].astype(int).tolist()
    val_idx = val_df['sample_id'].astype(int).tolist()

    train_idx_set = set(train_idx)
    val_idx_set = set(val_idx)

    if train_idx_set & val_idx_set:
        raise ValueError(f'Overlap between train and val indices for fold {fold}')
    if any(val_df['is_none_scaffold'].tolist()):
        raise ValueError(f'Validation set contains none-scaffold samples for fold {fold}')

    all_idx = train_idx_set | val_idx_set
    for idx in all_idx:
        if idx < 0 or idx >= total_samples:
            raise ValueError(f'Sample ID {idx} out of range [0, {total_samples - 1}]')

    return train_idx, val_idx


# ===========================# ===========================
# Molecular graph construction
# ===========================
def get_atom_features(atom: Chem.Atom) -> List[float]:
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetHybridization().real),
        float(atom.GetIsAromatic()),
        float(atom.GetMass()),
        float(atom.GetTotalNumHs()),
        float(atom.GetNumRadicalElectrons()),
        float(atom.GetIsotope()),
        float(atom.IsInRing()),
        float(atom.GetChiralTag()),
        float(atom.GetTotalValence()),
        float(atom.GetExplicitValence()),
        float(atom.GetImplicitValence()),
    ]


def get_bond_features(bond: Chem.Bond) -> List[float]:
    return [
        float(bond.GetBondType() == Chem.BondType.SINGLE),
        float(bond.GetBondType() == Chem.BondType.DOUBLE),
        float(bond.GetBondType() == Chem.BondType.TRIPLE),
        float(bond.GetIsAromatic()),
        float(bond.IsInRing()),
        float(bond.GetIsConjugated()),
        float(bond.GetStereo()),
    ]


def make_dummy_graph() -> Data:
    x = torch.zeros((1, 14), dtype=torch.float32)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, 7), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_pyg_data(smiles: str, use_dummy_on_failure: bool = True) -> Optional[Data]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return make_dummy_graph() if use_dummy_on_failure else None

        x = torch.tensor([get_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
        edge_indices: List[List[int]] = []
        edge_attrs: List[List[float]] = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_feat = get_bond_features(bond)
            edge_indices.extend([[i, j], [j, i]])
            edge_attrs.extend([bond_feat, bond_feat])

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 7), dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception:
        return make_dummy_graph() if use_dummy_on_failure else None


# =========================
# Data loading and datasets
# =========================
def load_aligned_multimodal_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Loading joint data from {data_dir}...', flush=True)
    train_csv = os.path.join(data_dir, 'multimodal_train.csv')
    rdkit_path = os.path.join(data_dir, 'rdkit3d_train.npy')
    xtb_path = os.path.join(data_dir, 'XTB_train.pth')

    train_df = pd.read_csv(train_csv)
    rdkit_features = np.load(rdkit_path)
    xtb_data = torch.load(xtb_path)
    xtb_smiles = xtb_data['smiles']
    xtb_features = xtb_data['features'].numpy()
    xtb_features_dict = dict(zip(xtb_smiles, xtb_features))

    if len(train_df) != len(rdkit_features):
        raise ValueError(f'CSV rows ({len(train_df)}) and RDKit rows ({len(rdkit_features)}) are not aligned.')

    aligned_targets: List[float] = []
    aligned_xtb: List[np.ndarray] = []
    aligned_rdkit: List[np.ndarray] = []
    aligned_smiles: List[str] = []
    skipped_missing_xtb = 0
    skipped_missing_target = 0

    for idx, row in train_df.iterrows():
        smiles = row['SMILES']
        target = row['MP']
        if pd.isna(smiles) or pd.isna(target):
            skipped_missing_target += 1
            continue
        if smiles not in xtb_features_dict:
            skipped_missing_xtb += 1
            continue
        aligned_smiles.append(smiles)
        aligned_targets.append(float(target))
        aligned_xtb.append(xtb_features_dict[smiles])
        aligned_rdkit.append(rdkit_features[idx])

    targets = np.asarray(aligned_targets, dtype=np.float32)
    xtb = np.asarray(aligned_xtb, dtype=np.float32)
    rdkit = np.asarray(aligned_rdkit, dtype=np.float32)

    print(f'Aligned samples: {len(aligned_smiles)}', flush=True)
    print(f'Skipped rows | missing target/smiles: {skipped_missing_target} | missing XTB: {skipped_missing_xtb}', flush=True)
    print(f'XTB shape: {xtb.shape} | RDKit shape: {rdkit.shape}', flush=True)
    return targets, xtb, rdkit, aligned_smiles


class JointMolGraphTextDataset(Dataset):
    def __init__(
        self,
        smiles_list: Sequence[str],
        xtb_features: np.ndarray,
        rdkit_features: np.ndarray,
        targets: np.ndarray,
        sample_ids: Optional[Sequence[int]] = None,
        cache_graphs: bool = True,
    ) -> None:
        self.smiles_list = list(smiles_list)
        self.xtb_features = np.asarray(xtb_features, dtype=np.float32)
        self.rdkit_features = np.asarray(rdkit_features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.sample_ids = np.asarray(sample_ids if sample_ids is not None else np.arange(len(self.targets)), dtype=np.int64)
        self.graphs: Optional[List[Data]] = None

        if cache_graphs:
            print(f'Building molecular graphs for {len(self.smiles_list)} samples...', flush=True)
            self.graphs = [smiles_to_pyg_data(smiles, use_dummy_on_failure=True) for smiles in self.smiles_list]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        graph = self.graphs[idx].clone() if self.graphs is not None else smiles_to_pyg_data(self.smiles_list[idx], use_dummy_on_failure=True)
        return {
            'sample_id': int(self.sample_ids[idx]),
            'smiles': self.smiles_list[idx],
            'graph': graph,
            'xtb': self.xtb_features[idx],
            'rdkit': self.rdkit_features[idx],
            'target': self.targets[idx],
        }


class JointCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        smiles_batch = [item['smiles'] for item in batch]
        tokenized = self.tokenizer(
            smiles_batch,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        graph_batch = Batch.from_data_list([item['graph'] for item in batch])
        xtb = torch.tensor(np.asarray([item['xtb'] for item in batch]), dtype=torch.float32)
        rdkit = torch.tensor(np.asarray([item['rdkit'] for item in batch]), dtype=torch.float32)
        targets = torch.tensor(np.asarray([item['target'] for item in batch]), dtype=torch.float32)
        sample_ids = torch.tensor(np.asarray([item['sample_id'] for item in batch]), dtype=torch.long)
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'graph_batch': graph_batch,
            'xtb': xtb,
            'rdkit': rdkit,
            'target': targets,
            'sample_id': sample_ids,
            'smiles': smiles_batch,
        }


# ========================
# MoLFormer encoder wrapper
# ========================
def _load_transformer(model_name: str) -> AutoModel:
    try:
        return AutoModel.from_pretrained(model_name, deterministic_eval=True, trust_remote_code=True)
    except TypeError:
        return AutoModel.from_pretrained(model_name, trust_remote_code=True)


class MolFormerEncoder(nn.Module):
    def __init__(self, model_name: str, enable_gradient_checkpointing: bool = True) -> None:
        super().__init__()
        self.encoder = _load_transformer(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.layer_list = self._resolve_layer_list()

        if enable_gradient_checkpointing and hasattr(self.encoder, 'gradient_checkpointing_enable'):
            try:
                self.encoder.gradient_checkpointing_enable()
            except Exception:
                pass

    def _resolve_layer_list(self) -> Optional[Sequence[nn.Module]]:
        candidates = [
            getattr(getattr(self.encoder, 'encoder', None), 'layer', None),
            getattr(getattr(self.encoder, 'model', None), 'layers', None),
            getattr(self.encoder, 'layers', None),
            getattr(getattr(self.encoder, 'transformer', None), 'layer', None),
            getattr(getattr(self.encoder, 'transformer', None), 'layers', None),
        ]
        for layers in candidates:
            if layers is not None:
                return layers
        return None

    def freeze_all_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n_layers: int) -> None:
        self.freeze_all_encoder()
        if n_layers <= 0 or self.layer_list is None:
            return
        for layer in list(self.layer_list)[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(outputs, 'pooler_output', None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0, :]
        return pooled


class DirectedMPNNEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int = 14,
        edge_dim: int = 7,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.W_i = nn.Linear(node_dim + edge_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.W_o = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.attn_pool = GlobalAttention(self.gate_nn)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = x.float()
        edge_attr = edge_attr.float()
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            m_v_empty = torch.zeros(num_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
            node_h = self.W_o(torch.cat([x, m_v_empty], dim=1))
            x_attn = self.attn_pool(node_h, batch)
            x_combined = torch.cat([x_attn, global_max_pool(node_h, batch)], dim=1)
            return self.output_proj(x_combined)

        src, dst = edge_index
        rev_indices = torch.arange(num_edges, device=x.device)
        rev_indices[0::2] += 1
        rev_indices[1::2] -= 1

        h_vw = self.W_i(torch.cat([x[src], edge_attr], dim=1))
        h_vw = nn.functional.silu(h_vw)

        for _ in range(self.num_layers):
            m_w = torch.zeros(num_nodes, self.hidden_dim, device=x.device, dtype=h_vw.dtype)
            m_w.index_add_(0, dst, h_vw)
            m_v_to_w = m_w[src]
            h_wv = h_vw[rev_indices]
            m_vw = m_v_to_w - h_wv
            h_vw = self.gru(m_vw, h_vw)
            h_vw = self.dropout(h_vw)

        m_v_final = torch.zeros(num_nodes, self.hidden_dim, device=x.device, dtype=h_vw.dtype)
        m_v_final.index_add_(0, dst, h_vw)
        node_h = self.W_o(torch.cat([x, m_v_final], dim=1))
        x_attn = self.attn_pool(node_h, batch)
        x_combined = torch.cat([x_attn, global_max_pool(node_h, batch)], dim=1)
        return self.output_proj(x_combined)


class DynamicReadoutFusionHead(nn.Module):
    def __init__(
        self,
        bert_dim: int,
        dmpnn_dim: int,
        xtb_dim: int,
        rdkit_dim: int,
        common_hidden_dim: int = 192,
        xtb_hidden_dim: int = 128,
        dropout: float = 0.325,
        use_rdkit_in_xtb: bool = True,
        xtb_depth: int = 3,
        readout_dim: int = 16,
        dynamic_weight_scale: float = 1.0,
        bias_hidden_dim: int = 64,
        main_branch_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_rdkit_in_xtb = use_rdkit_in_xtb
        self.dynamic_weight_scale = dynamic_weight_scale
        self.main_branch_noise_std = main_branch_noise_std

        self.proj_bert = nn.Sequential(
            nn.Linear(bert_dim, common_hidden_dim),
            nn.LayerNorm(common_hidden_dim),
            nn.GELU(),
        )
        self.proj_dmpnn = nn.Sequential(
            nn.Linear(dmpnn_dim, common_hidden_dim),
            nn.LayerNorm(common_hidden_dim),
            nn.GELU(),
        )
        self.main_head = nn.Sequential(
            nn.Linear(common_hidden_dim * 2, common_hidden_dim),
            nn.LayerNorm(common_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(common_hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, readout_dim),
        )

        xtb_in_dim = xtb_dim + (rdkit_dim if self.use_rdkit_in_xtb else 0)
        xtb_layers_seq: List[nn.Module] = []
        current_dim = xtb_in_dim
        for _ in range(xtb_depth):
            xtb_layers_seq.extend([
                nn.Linear(current_dim, xtb_hidden_dim),
                nn.LayerNorm(xtb_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = xtb_hidden_dim
        if xtb_layers_seq:
            xtb_layers_seq = xtb_layers_seq[:-1]
        self.xtb_encoder = nn.Sequential(*xtb_layers_seq)

        self.weight_head = nn.Sequential(
            nn.Linear(xtb_hidden_dim, bias_hidden_dim),
            nn.LayerNorm(bias_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(bias_hidden_dim, readout_dim),
        )
        self.bias_head = nn.Sequential(
            nn.Linear(xtb_hidden_dim, bias_hidden_dim),
            nn.LayerNorm(bias_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(bias_hidden_dim, 1),
        )

    def forward(
        self,
        bert_feat: torch.Tensor,
        dmpnn_feat: torch.Tensor,
        xtb_feat: torch.Tensor,
        rdkit_feat: torch.Tensor,
        return_parts: bool = False,
    ) -> torch.Tensor:
        h_bert = self.proj_bert(bert_feat)
        h_dmpnn = self.proj_dmpnn(dmpnn_feat)
        h_main_in = torch.cat([h_bert, h_dmpnn], dim=-1)
        z_main = self.main_head(h_main_in)
        z_main = nn.functional.layer_norm(z_main, (z_main.size(-1),))
        if self.training and self.main_branch_noise_std > 0:
            z_main = z_main + torch.randn_like(z_main) * self.main_branch_noise_std

        h_xtb_in = torch.cat([xtb_feat, rdkit_feat], dim=-1) if self.use_rdkit_in_xtb else xtb_feat
        h_xtb = self.xtb_encoder(h_xtb_in)
        w_xtb = self.dynamic_weight_scale * torch.tanh(self.weight_head(h_xtb))
        main_pred = torch.sum(z_main * w_xtb, dim=-1)
        bias_pred = self.bias_head(h_xtb).squeeze(-1)
        pred = main_pred + bias_pred

        if return_parts:
            return {
                'pred': pred,
                'z_main_abs': z_main.abs().mean().detach(),
                'h_xtb_abs': h_xtb.abs().mean().detach(),
                'w_xtb_abs': w_xtb.abs().mean().detach(),
                'main_pred_abs': main_pred.abs().mean().detach(),
                'bias_pred_abs': bias_pred.abs().mean().detach(),
                'pred_abs': pred.abs().mean().detach(),
                'main_pred': main_pred.detach(),
                'bias_pred': bias_pred.detach(),
                'w_xtb_abs_mean_per_sample': w_xtb.abs().mean(dim=-1).detach(),
                'z_main_l2_per_sample': torch.norm(z_main, dim=-1).detach(),
                'h_xtb_l2_per_sample': torch.norm(h_xtb, dim=-1).detach(),
            }
        return pred


class JointMolFormerDMPNNXTBModel(nn.Module):
    def __init__(self, cfg: TrainConfig, xtb_dim: int, rdkit_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.bert_encoder = MolFormerEncoder(cfg.model_name, enable_gradient_checkpointing=True)
        self.dmpnn_encoder = DirectedMPNNEncoder(
            hidden_dim=cfg.dmpnn_hidden_dim,
            output_dim=cfg.dmpnn_output_dim,
            num_layers=cfg.dmpnn_layers,
            dropout=cfg.dmpnn_dropout,
        )
        self.fusion_head = DynamicReadoutFusionHead(
            bert_dim=self.bert_encoder.hidden_size,
            dmpnn_dim=cfg.dmpnn_output_dim,
            xtb_dim=xtb_dim,
            rdkit_dim=rdkit_dim,
            common_hidden_dim=cfg.common_hidden_dim,
            xtb_hidden_dim=cfg.xtb_hidden_dim,
            dropout=cfg.dropout,
            use_rdkit_in_xtb=cfg.use_rdkit_in_xtb,
            xtb_depth=cfg.xtb_depth,
            readout_dim=cfg.readout_dim,
            dynamic_weight_scale=cfg.dynamic_weight_scale,
            bias_hidden_dim=cfg.bias_hidden_dim,
            main_branch_noise_std=cfg.main_branch_noise_std,
        )
        self.current_unfrozen_bert_layers: int = 0
        self.set_bert_trainability(0)

    def set_bert_trainability(self, unfreeze_last_n_layers: int) -> None:
        self.current_unfrozen_bert_layers = max(0, unfreeze_last_n_layers)
        if self.current_unfrozen_bert_layers <= 0:
            self.bert_encoder.freeze_all_encoder()
        else:
            self.bert_encoder.unfreeze_last_n_layers(self.current_unfrozen_bert_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_batch: Batch,
        xtb_feat: torch.Tensor,
        rdkit_feat: torch.Tensor,
        return_parts: bool = False,
    ) -> torch.Tensor:
        bert_feat = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        dmpnn_feat = self.dmpnn_encoder(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch)
        return self.fusion_head(bert_feat, dmpnn_feat, xtb_feat, rdkit_feat, return_parts=return_parts)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build_optimizer(model: JointMolFormerDMPNNXTBModel, cfg: TrainConfig) -> optim.Optimizer:
    xtb_base_wd = cfg.fusion_weight_decay * cfg.xtb_wd_mult
    param_groups = [
        {
            'group_name': 'bert_encoder',
            'params': list(model.bert_encoder.encoder.parameters()),
            'lr': 0.0,
            'weight_decay': cfg.bert_encoder_weight_decay,
            'base_weight_decay': cfg.bert_encoder_weight_decay,
        },
        {
            'group_name': 'bert_projection',
            'params': list(model.fusion_head.proj_bert.parameters()),
            'lr': cfg.bert_projection_lr,
            'weight_decay': cfg.bert_projection_weight_decay,
            'base_weight_decay': cfg.bert_projection_weight_decay,
        },
        {
            'group_name': 'dmpnn',
            'params': list(model.dmpnn_encoder.parameters()),
            'lr': cfg.dmpnn_lr,
            'weight_decay': cfg.dmpnn_weight_decay,
            'base_weight_decay': cfg.dmpnn_weight_decay,
        },
        {
            'group_name': 'common_fusion',
            'params': list(model.fusion_head.proj_dmpnn.parameters()) + list(model.fusion_head.main_head.parameters()),
            'lr': cfg.fusion_lr,
            'weight_decay': cfg.fusion_weight_decay,
            'base_weight_decay': cfg.fusion_weight_decay,
        },
        {
            'group_name': 'xtb_encoder',
            'params': list(model.fusion_head.xtb_encoder.parameters()),
            'lr': cfg.fusion_lr,
            'weight_decay': xtb_base_wd,
            'base_weight_decay': xtb_base_wd,
        },
        {
            'group_name': 'weight_head',
            'params': list(model.fusion_head.weight_head.parameters()),
            'lr': cfg.fusion_lr,
            'weight_decay': xtb_base_wd,
            'base_weight_decay': xtb_base_wd,
        },
        {
            'group_name': 'bias_head',
            'params': list(model.fusion_head.bias_head.parameters()),
            'lr': cfg.fusion_lr,
            'weight_decay': xtb_base_wd,
            'base_weight_decay': xtb_base_wd,
        },
    ]
    return optim.AdamW(param_groups)


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def maybe_freeze_main_head(model: JointMolFormerDMPNNXTBModel, epoch: int, cfg: TrainConfig) -> bool:
    if epoch == cfg.freeze_main_head_epoch:
        freeze_module(model.fusion_head.main_head)
        return True
    return False


def apply_regularization_schedule(optimizer: optim.Optimizer, epoch: int, cfg: TrainConfig) -> bool:
    boosted = epoch >= cfg.reg_boost_epoch
    changed = False
    xtb_groups = {'xtb_encoder', 'weight_head', 'bias_head'}
    for group in optimizer.param_groups:
        base_wd = group.get('base_weight_decay', group.get('weight_decay', 0.0))
        target_wd = base_wd * (cfg.reg_boost_factor if boosted and group.get('group_name') in xtb_groups else 1.0)
        if group.get('weight_decay') != target_wd:
            group['weight_decay'] = target_wd
            changed = True
    return changed


def apply_epoch_stage(model: JointMolFormerDMPNNXTBModel, optimizer: optim.Optimizer, epoch: int, cfg: TrainConfig) -> str:
    polish_start = max(cfg.freeze_bert_epochs + 1, cfg.max_epochs - cfg.final_tune_epochs + 1)
    if epoch <= cfg.freeze_bert_epochs:
        stage = 'warmup_frozen_bert'
        model.set_bert_trainability(0)
        lr_map = {
            'bert_encoder': 0.0,
            'bert_projection': cfg.bert_projection_lr,
            'dmpnn': cfg.dmpnn_lr,
            'common_fusion': cfg.fusion_lr,
            'xtb_encoder': cfg.fusion_lr,
            'weight_head': cfg.fusion_lr,
            'bias_head': cfg.fusion_lr,
        }
    elif epoch < polish_start:
        stage = 'joint_main'
        model.set_bert_trainability(cfg.bert_unfreeze_layers)
        lr_map = {
            'bert_encoder': cfg.bert_encoder_lr,
            'bert_projection': cfg.bert_projection_lr,
            'dmpnn': cfg.dmpnn_lr,
            'common_fusion': cfg.fusion_lr,
            'xtb_encoder': cfg.fusion_lr,
            'weight_head': cfg.fusion_lr,
            'bias_head': cfg.fusion_lr,
        }
    else:
        stage = 'joint_polish'
        model.set_bert_trainability(cfg.bert_unfreeze_layers)
        lr_map = {
            'bert_encoder': cfg.bert_encoder_lr * cfg.final_lr_scale,
            'bert_projection': cfg.bert_projection_lr * cfg.final_lr_scale,
            'dmpnn': cfg.dmpnn_lr * cfg.final_lr_scale,
            'common_fusion': cfg.fusion_lr * cfg.final_lr_scale,
            'xtb_encoder': cfg.fusion_lr * cfg.final_lr_scale,
            'weight_head': cfg.fusion_lr * cfg.final_lr_scale,
            'bias_head': cfg.fusion_lr * cfg.final_lr_scale,
        }

    for group in optimizer.param_groups:
        group['lr'] = lr_map[group['group_name']]
    return stage


def get_group_lrs(optimizer: optim.Optimizer) -> Dict[str, float]:
    return {group['group_name']: group['lr'] for group in optimizer.param_groups}


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        'input_ids': batch['input_ids'].to(device, non_blocking=True),
        'attention_mask': batch['attention_mask'].to(device, non_blocking=True),
        'graph_batch': batch['graph_batch'].to(device),
        'xtb': batch['xtb'].to(device, non_blocking=True),
        'rdkit': batch['rdkit'].to(device, non_blocking=True),
        'target': batch['target'].to(device, non_blocking=True),
        'sample_id': batch['sample_id'].to(device, non_blocking=True),
        'smiles': batch['smiles'],
    }


def build_fold_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    collator: JointCollator,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collator,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: JointMolFormerDMPNNXTBModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    cfg: TrainConfig,
    bert_encoder_trainable: bool,
) -> Dict[str, float]:
    model.train()
    if not bert_encoder_trainable:
        model.bert_encoder.encoder.eval()

    total_loss = 0.0
    total_mae_std = 0.0
    total_z_main_abs = 0.0
    total_h_xtb_abs = 0.0
    total_w_xtb_abs = 0.0
    total_main_pred_abs = 0.0
    total_bias_pred_abs = 0.0
    total_pred_abs = 0.0
    total_count = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        batch = _move_batch_to_device(batch, device)
        batch_size = batch['target'].size(0)

        parts = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            graph_batch=batch['graph_batch'],
            xtb_feat=batch['xtb'],
            rdkit_feat=batch['rdkit'],
            return_parts=True,
        )
        pred = parts['pred']
        loss = criterion(pred, batch['target']) / cfg.grad_accum_steps
        loss.backward()

        if step % cfg.grad_accum_steps == 0 or step == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * cfg.grad_accum_steps * batch_size
        total_mae_std += torch.abs(pred - batch['target']).sum().item()
        total_z_main_abs += parts['z_main_abs'].item() * batch_size
        total_h_xtb_abs += parts['h_xtb_abs'].item() * batch_size
        total_w_xtb_abs += parts['w_xtb_abs'].item() * batch_size
        total_main_pred_abs += parts['main_pred_abs'].item() * batch_size
        total_bias_pred_abs += parts['bias_pred_abs'].item() * batch_size
        total_pred_abs += parts['pred_abs'].item() * batch_size
        total_count += batch_size

    return {
        'loss': total_loss / total_count,
        'mae_std': total_mae_std / total_count,
        'z_main_abs': total_z_main_abs / total_count,
        'h_xtb_abs': total_h_xtb_abs / total_count,
        'w_xtb_abs': total_w_xtb_abs / total_count,
        'main_pred_abs': total_main_pred_abs / total_count,
        'bias_pred_abs': total_bias_pred_abs / total_count,
        'pred_abs': total_pred_abs / total_count,
    }


@torch.no_grad()
def validate(
    model: JointMolFormerDMPNNXTBModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler_t: StandardScaler,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_mae_std = 0.0
    total_z_main_abs = 0.0
    total_h_xtb_abs = 0.0
    total_w_xtb_abs = 0.0
    total_main_pred_abs = 0.0
    total_bias_pred_abs = 0.0
    total_pred_abs = 0.0
    total_count = 0
    pred_std_list: List[np.ndarray] = []
    target_std_list: List[np.ndarray] = []
    sample_id_list: List[np.ndarray] = []
    smiles_all: List[str] = []
    main_pred_std_list: List[np.ndarray] = []
    bias_pred_std_list: List[np.ndarray] = []
    w_xtb_abs_mean_list: List[np.ndarray] = []
    z_main_l2_list: List[np.ndarray] = []
    h_xtb_l2_list: List[np.ndarray] = []

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        batch_size = batch['target'].size(0)
        parts = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            graph_batch=batch['graph_batch'],
            xtb_feat=batch['xtb'],
            rdkit_feat=batch['rdkit'],
            return_parts=True,
        )
        pred = parts['pred']
        loss = criterion(pred, batch['target'])

        total_loss += loss.item() * batch_size
        total_mae_std += torch.abs(pred - batch['target']).sum().item()
        total_z_main_abs += parts['z_main_abs'].item() * batch_size
        total_h_xtb_abs += parts['h_xtb_abs'].item() * batch_size
        total_w_xtb_abs += parts['w_xtb_abs'].item() * batch_size
        total_main_pred_abs += parts['main_pred_abs'].item() * batch_size
        total_bias_pred_abs += parts['bias_pred_abs'].item() * batch_size
        total_pred_abs += parts['pred_abs'].item() * batch_size
        total_count += batch_size
        pred_std_list.append(pred.cpu().numpy())
        target_std_list.append(batch['target'].cpu().numpy())
        sample_id_list.append(batch['sample_id'].cpu().numpy())
        smiles_all.extend(batch['smiles'])
        main_pred_std_list.append(parts['main_pred'].cpu().numpy())
        bias_pred_std_list.append(parts['bias_pred'].cpu().numpy())
        w_xtb_abs_mean_list.append(parts['w_xtb_abs_mean_per_sample'].cpu().numpy())
        z_main_l2_list.append(parts['z_main_l2_per_sample'].cpu().numpy())
        h_xtb_l2_list.append(parts['h_xtb_l2_per_sample'].cpu().numpy())

    pred_std = np.concatenate(pred_std_list, axis=0)
    target_std = np.concatenate(target_std_list, axis=0)
    sample_ids = np.concatenate(sample_id_list, axis=0).astype(np.int64)
    pred_raw = scaler_t.inverse_transform(pred_std.reshape(-1, 1)).flatten()
    target_raw = scaler_t.inverse_transform(target_std.reshape(-1, 1)).flatten()
    mae_raw = float(np.mean(np.abs(pred_raw - target_raw)))

    order = np.argsort(sample_ids)
    sample_ids = sample_ids[order]
    pred_std = pred_std[order]
    target_std = target_std[order]
    pred_raw = pred_raw[order]
    target_raw = target_raw[order]
    main_pred_std = np.concatenate(main_pred_std_list, axis=0)[order]
    bias_pred_std = np.concatenate(bias_pred_std_list, axis=0)[order]
    w_xtb_abs_mean = np.concatenate(w_xtb_abs_mean_list, axis=0)[order]
    z_main_l2 = np.concatenate(z_main_l2_list, axis=0)[order]
    h_xtb_l2 = np.concatenate(h_xtb_l2_list, axis=0)[order]
    smiles_sorted = [smiles_all[i] for i in order.tolist()]

    return {
        'loss': total_loss / total_count,
        'mae_std': total_mae_std / total_count,
        'mae_raw': mae_raw,
        'z_main_abs': total_z_main_abs / total_count,
        'h_xtb_abs': total_h_xtb_abs / total_count,
        'w_xtb_abs': total_w_xtb_abs / total_count,
        'main_pred_abs': total_main_pred_abs / total_count,
        'bias_pred_abs': total_bias_pred_abs / total_count,
        'pred_abs': total_pred_abs / total_count,
        'sample_ids': sample_ids,
        'smiles': smiles_sorted,
        'pred_std': pred_std,
        'target_std': target_std,
        'pred_raw': pred_raw,
        'target_raw': target_raw.astype(np.float32),
        'main_pred_std': main_pred_std,
        'bias_pred_std': bias_pred_std,
        'w_xtb_abs_mean': w_xtb_abs_mean,
        'z_main_l2': z_main_l2,
        'h_xtb_l2': h_xtb_l2,
    }


def build_prediction_frame(
    run_id: str,
    fold: int,
    selected_epoch: int,
    split_manifest_lookup: pd.DataFrame,
    val_stats: Dict[str, Any],
) -> pd.DataFrame:
    sample_ids = [int(x) for x in np.asarray(val_stats['sample_ids']).tolist()]
    meta = split_manifest_lookup.loc[sample_ids].reset_index(drop=True).copy()
    if len(meta) != len(val_stats['pred_raw']):
        raise ValueError('Prediction rows do not match manifest rows.')

    smiles_from_loader = list(val_stats.get('smiles', []))
    if smiles_from_loader and smiles_from_loader != meta['smiles'].tolist():
        meta['smiles_loader'] = smiles_from_loader

    pred_df = meta.copy()
    pred_df['run_id'] = run_id
    pred_df['fold'] = fold
    pred_df['selected_epoch'] = selected_epoch
    pred_df['split'] = 'val'
    pred_df['target_std'] = np.asarray(val_stats['target_std'], dtype=np.float32)
    pred_df['pred_std'] = np.asarray(val_stats['pred_std'], dtype=np.float32)
    pred_df['target_raw'] = np.asarray(val_stats['target_raw'], dtype=np.float32)
    pred_df['pred_raw'] = np.asarray(val_stats['pred_raw'], dtype=np.float32)
    pred_df['error_raw'] = pred_df['pred_raw'] - pred_df['target_raw']
    pred_df['abs_error'] = pred_df['error_raw'].abs()
    pred_df['sq_error'] = pred_df['error_raw'] ** 2
    pred_df['main_pred_std'] = np.asarray(val_stats['main_pred_std'], dtype=np.float32)
    pred_df['bias_pred_std'] = np.asarray(val_stats['bias_pred_std'], dtype=np.float32)
    pred_df['w_xtb_abs_mean'] = np.asarray(val_stats['w_xtb_abs_mean'], dtype=np.float32)
    pred_df['z_main_l2'] = np.asarray(val_stats['z_main_l2'], dtype=np.float32)
    pred_df['h_xtb_l2'] = np.asarray(val_stats['h_xtb_l2'], dtype=np.float32)
    return pred_df


def build_error_bin_summary(oof_df: pd.DataFrame, bin_width: int = 50) -> pd.DataFrame:
    if oof_df.empty:
        return pd.DataFrame(columns=['target_bin', 'count', 'mae', 'rmse', 'mean_target_raw', 'mean_pred_raw'])

    max_target = float(oof_df['target_raw'].max())
    upper = max(bin_width, int(math.ceil(max(max_target, 0.0) / bin_width)) * bin_width)
    bins = [-np.inf, 0.0] + [float(edge) for edge in range(bin_width, upper + bin_width, bin_width)] + [np.inf]
    labels = ['<0'] + [f'{left}-{right}' for left, right in zip(range(0, upper, bin_width), range(bin_width, upper + bin_width, bin_width))] + [f'>={upper}']

    working = oof_df.copy()
    working['target_bin'] = pd.cut(working['target_raw'], bins=bins, labels=labels, right=False, include_lowest=True)
    grouped = working.groupby('target_bin', observed=False)
    summary = grouped.agg(
        count=('sample_id', 'size'),
        mae=('abs_error', 'mean'),
        mean_sq_error=('sq_error', 'mean'),
        mean_target_raw=('target_raw', 'mean'),
        mean_pred_raw=('pred_raw', 'mean'),
    ).reset_index()
    summary['rmse'] = np.sqrt(summary['mean_sq_error'])
    return summary.drop(columns=['mean_sq_error'])


def save_best_checkpoint(
    model: JointMolFormerDMPNNXTBModel,
    output_dir: str,
    fold: int,
    epoch: int,
    best_mae: float,
    cfg: TrainConfig,
    scaler_target: StandardScaler,
    optimizer: Optional[optim.Optimizer] = None,
) -> str:
    full_path = os.path.join(output_dir, f'best_fold{fold}_full_model.pt')
    checkpoint_data = {
        'checkpoint_format_version': 2,
        'checkpoint_type': 'full_model',
        'epoch': epoch,
        'best_val_mae_raw': best_mae,
        'config': asdict(cfg),
        'target_scaler_mean': float(scaler_target.mean_[0]),
        'target_scaler_scale': float(scaler_target.scale_[0]),
        'state_dict': model.state_dict(),
    }
    if optimizer is not None:
        checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint_data, full_path)
    stable_path = os.path.join(output_dir, f'best_fold{fold}.pt')
    shutil.copy2(full_path, stable_path)
    return full_path


def plot_fold_curves(history: Dict[str, List[float]], fold: int, output_dir: str) -> None:
    epochs = history['epoch']

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold} Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fold{fold}_loss_curve.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['val_mae_raw'], label='Val MAE(raw)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE(raw)')
    plt.title(f'Fold {fold} Val MAE(raw)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fold{fold}_val_mae_raw_curve.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_mae_std'], label='Train MAE(std)')
    plt.plot(epochs, history['val_mae_std'], label='Val MAE(std)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE(std)')
    plt.title(f'Fold {fold} MAE(std) Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fold{fold}_mae_std_curve.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_z_main_abs'], label='Train |z_main|')
    plt.plot(epochs, history['val_z_main_abs'], label='Val |z_main|')
    plt.plot(epochs, history['train_h_xtb_abs'], label='Train |h_xtb|')
    plt.plot(epochs, history['val_h_xtb_abs'], label='Val |h_xtb|')
    plt.xlabel('Epoch')
    plt.ylabel('Average absolute magnitude')
    plt.title(f'Fold {fold} Hidden Magnitude Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fold{fold}_hidden_magnitude_curve.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_main_pred_abs'], label='Train |main_pred|')
    plt.plot(epochs, history['val_main_pred_abs'], label='Val |main_pred|')
    plt.plot(epochs, history['train_bias_pred_abs'], label='Train |b_pred|')
    plt.plot(epochs, history['val_bias_pred_abs'], label='Val |b_pred|')
    plt.plot(epochs, history['train_w_xtb_abs'], label='Train |w_xtb|')
    plt.plot(epochs, history['val_w_xtb_abs'], label='Val |w_xtb|')
    plt.xlabel('Epoch')
    plt.ylabel('Average absolute prediction')
    plt.title(f'Fold {fold} Residual Boosting Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fold{fold}_residual_boosting_curve.png'), dpi=200)
    plt.close()


def plot_summary(all_fold_best_mae: Sequence[float], output_dir: str, selected_folds: Sequence[int]) -> None:
    plt.figure(figsize=(7, 5))
    folds = np.asarray(selected_folds)
    plt.plot(folds, all_fold_best_mae, marker='o')
    plt.xticks(folds)
    plt.xlabel('Fold')
    plt.ylabel('Best Val MAE(raw)')
    plt.title('Best Validation MAE(raw) by Fold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fold_best_mae_summary.png'), dpi=200)
    plt.close()


def run_training(cfg: TrainConfig, device: torch.device) -> None:
    set_seed(cfg.seed)
    selected_folds = parse_selected_folds(cfg)
    if not selected_folds:
        raise ValueError('No folds selected to run.')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_basename = get_run_basename(timestamp, cfg)
    output_dir = os.path.join(cfg.outputs_root, run_basename)
    run_id = run_basename
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), os.path.join(output_dir, os.path.basename(__file__)))
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Clean residual boosting training script started', flush=True)
    print(f'Output dir: {output_dir}', flush=True)
    print(f'Using device: {device}', flush=True)
    save_json(asdict(cfg), os.path.join(output_dir, 'config.json'))

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    git_commit = get_git_commit(repo_dir)

    targets_raw, xtb_raw, rdkit_raw, smiles_list = load_aligned_multimodal_data(cfg.data_dir)
    total_sample_count = len(smiles_list)

    split_dir = os.path.abspath(cfg.split_dir)
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Using frozen scaffold split from {split_dir}', flush=True)
    split_manifest = load_frozen_split_manifest(split_dir)
    validate_frozen_manifest_alignment(smiles_list, split_manifest)
    split_manifest_lookup = split_manifest.set_index('sample_id', drop=False)

    none_idx = split_manifest[split_manifest['is_none_scaffold']]['sample_id'].astype(int).tolist()
    summary_file = os.path.join(split_dir, 'split_summary.json')
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            frozen_split_summary = json.load(f)
    else:
        fold_val_sizes = {
            f'fold_{fold}': int((split_manifest['assigned_val_fold'] == fold).sum())
            for fold in range(1, cfg.n_folds + 1)
        }
        frozen_split_summary = {
            'total_aligned_samples': total_sample_count,
            'valid_scaffold_samples': int((~split_manifest['is_none_scaffold']).sum()),
            'none_scaffold_samples_train_only': len(none_idx),
            'unique_valid_scaffolds': None,
            'fold_valid_scaffold_val_sizes': fold_val_sizes,
            'n_folds': cfg.n_folds,
        }

    split_summary = {
        'run_id': run_id,
        'split_source': 'frozen_scaffold',
        'split_dir': split_dir,
        'total_aligned_samples': int(frozen_split_summary.get('total_aligned_samples', total_sample_count)),
        'valid_scaffold_samples': int(
            frozen_split_summary.get(
                'valid_scaffold_samples',
                int((~split_manifest['is_none_scaffold']).sum()),
            )
        ),
        'none_scaffold_samples_train_only': int(
            frozen_split_summary.get('none_scaffold_samples_train_only', len(none_idx))
        ),
        'unique_valid_scaffolds': frozen_split_summary.get('unique_valid_scaffolds'),
        'fold_valid_scaffold_val_sizes': frozen_split_summary.get('fold_valid_scaffold_val_sizes', {}),
        'folds_to_run': selected_folds,
    }
    valid_scaffold_sample_count = split_summary['valid_scaffold_samples']

    save_json(split_summary, os.path.join(output_dir, 'split_summary.json'))
    save_dataframe(split_manifest, os.path.join(output_dir, 'split_manifest'), index=False)
    print(json.dumps(split_summary, indent=2, ensure_ascii=False), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    collator = JointCollator(tokenizer=tokenizer, max_length=cfg.max_length)

    none_idx_set = set(none_idx)
    all_fold_best_mae: List[float] = []
    all_fold_records: List[Dict[str, Any]] = []
    all_oof_frames: List[pd.DataFrame] = []

    for fold in selected_folds:
        train_idx, val_idx = load_frozen_fold_indices(split_dir, fold, total_sample_count)
        val_idx_set = set(val_idx)
        val_is_none_scaffold = split_manifest.loc[val_idx]['is_none_scaffold'].tolist()
        assert not any(val_is_none_scaffold), 'None-scaffold samples leaked into validation set.'
        assert len(set(train_idx) & val_idx_set) == 0, 'Train/Val overlap detected.'
        assert none_idx_set.issubset(set(train_idx)), 'Some none-scaffold samples are missing from the frozen train split.'

        print('\n' + '=' * 80, flush=True)
        print(
            f'Fold {fold}/{cfg.n_folds} | Split: frozen_scaffold | '
            f'Train: {len(train_idx)} | Val: {len(val_idx)} | None(train-only): {len(none_idx)}',
            flush=True,
        )
        print('=' * 80, flush=True)

        fold_scaler_dir = os.path.join(output_dir, f'scalers_fold{fold}')
        imp_xtb = SimpleImputer(strategy='median')
        imp_rdkit = SimpleImputer(strategy='median')
        scaler_xtb = StandardScaler()
        scaler_rdkit = StandardScaler()
        scaler_t = StandardScaler()

        xtb_train = scaler_xtb.fit_transform(imp_xtb.fit_transform(xtb_raw[train_idx]))
        xtb_val = scaler_xtb.transform(imp_xtb.transform(xtb_raw[val_idx]))
        rdkit_train = scaler_rdkit.fit_transform(imp_rdkit.fit_transform(rdkit_raw[train_idx]))
        rdkit_val = scaler_rdkit.transform(imp_rdkit.transform(rdkit_raw[val_idx]))
        t_train = scaler_t.fit_transform(targets_raw[train_idx].reshape(-1, 1)).flatten().astype(np.float32)
        t_val = scaler_t.transform(targets_raw[val_idx].reshape(-1, 1)).flatten().astype(np.float32)

        os.makedirs(fold_scaler_dir, exist_ok=True)
        joblib.dump(imp_xtb, os.path.join(fold_scaler_dir, 'xtb_imputer.pkl'))
        joblib.dump(imp_rdkit, os.path.join(fold_scaler_dir, 'rdkit_imputer.pkl'))
        joblib.dump(scaler_xtb, os.path.join(fold_scaler_dir, 'xtb_scaler.pkl'))
        joblib.dump(scaler_rdkit, os.path.join(fold_scaler_dir, 'rdkit_scaler.pkl'))
        joblib.dump(scaler_t, os.path.join(fold_scaler_dir, 'target_scaler.pkl'))

        train_dataset = JointMolGraphTextDataset(
            smiles_list=[smiles_list[idx] for idx in train_idx],
            xtb_features=xtb_train,
            rdkit_features=rdkit_train,
            targets=t_train,
            sample_ids=train_idx,
            cache_graphs=cfg.cache_graphs,
        )
        val_dataset = JointMolGraphTextDataset(
            smiles_list=[smiles_list[idx] for idx in val_idx],
            xtb_features=xtb_val,
            rdkit_features=rdkit_val,
            targets=t_val,
            sample_ids=val_idx,
            cache_graphs=cfg.cache_graphs,
        )

        current_batch_size = cfg.batch_size
        train_loader, val_loader = build_fold_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collator=collator,
            device=device,
            batch_size=current_batch_size,
            num_workers=cfg.num_workers,
        )

        model = JointMolFormerDMPNNXTBModel(cfg=cfg, xtb_dim=xtb_train.shape[1], rdkit_dim=rdkit_train.shape[1]).to(device)

        if fold == selected_folds[0]:
            print('Parameter counts (excluding MoLFormer backbone shown separately):', flush=True)
            print(f'  MoLFormer backbone: {count_parameters(model.bert_encoder.encoder):,}', flush=True)
            print(f'  BERT projection: {count_parameters(model.fusion_head.proj_bert):,}', flush=True)
            print(f'  D-MPNN encoder: {count_parameters(model.dmpnn_encoder):,}', flush=True)
            print(f'  D-MPNN projection: {count_parameters(model.fusion_head.proj_dmpnn):,}', flush=True)
            print(f'  Main head: {count_parameters(model.fusion_head.main_head):,}', flush=True)
            print(f'  XTB encoder: {count_parameters(model.fusion_head.xtb_encoder):,}', flush=True)
            print(f'  Weight head: {count_parameters(model.fusion_head.weight_head):,}', flush=True)
            print(f'  Bias head: {count_parameters(model.fusion_head.bias_head):,}', flush=True)
            non_backbone = (
                count_parameters(model.fusion_head.proj_bert)
                + count_parameters(model.dmpnn_encoder)
                + count_parameters(model.fusion_head.proj_dmpnn)
                + count_parameters(model.fusion_head.main_head)
                + count_parameters(model.fusion_head.xtb_encoder)
                + count_parameters(model.fusion_head.weight_head)
                + count_parameters(model.fusion_head.bias_head)
            )
            print(f'  Total excluding MoLFormer backbone: {non_backbone:,}', flush=True)

        optimizer = build_optimizer(model, cfg)
        criterion = nn.HuberLoss(delta=cfg.huber_delta)

        best_mae = float('inf')
        best_epoch = -1
        best_stage = ''
        best_lrs: Dict[str, float] = {}
        best_common_wd = None
        best_xtb_wd = None
        best_checkpoint_path = ''
        best_val_prediction_frame: Optional[pd.DataFrame] = None

        history: Dict[str, List[Any]] = {
            'run_id': [],
            'fold': [],
            'epoch': [],
            'stage': [],
            'train_loss': [],
            'val_loss': [],
            'train_mae_std': [],
            'val_mae_std': [],
            'val_mae_raw': [],
            'best_so_far_val_mae_raw': [],
            'train_z_main_abs': [],
            'val_z_main_abs': [],
            'train_h_xtb_abs': [],
            'val_h_xtb_abs': [],
            'train_w_xtb_abs': [],
            'val_w_xtb_abs': [],
            'train_main_pred_abs': [],
            'val_main_pred_abs': [],
            'train_bias_pred_abs': [],
            'val_bias_pred_abs': [],
            'train_pred_abs': [],
            'val_pred_abs': [],
            'batch_size': [],
            'bert_encoder_lr': [],
            'bert_projection_lr': [],
            'dmpnn_lr': [],
            'fusion_lr': [],
            'xtb_lr': [],
            'common_weight_decay': [],
            'xtb_weight_decay': [],
        }

        for epoch in range(1, cfg.max_epochs + 1):
            stage = apply_epoch_stage(model, optimizer, epoch, cfg)
            main_head_frozen_now = maybe_freeze_main_head(model, epoch, cfg)
            reg_changed = apply_regularization_schedule(optimizer, epoch, cfg)
            if epoch == cfg.stage2_start_epoch and cfg.stage2_batch_size != current_batch_size:
                current_batch_size = cfg.stage2_batch_size
                train_loader, val_loader = build_fold_loaders(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    collator=collator,
                    device=device,
                    batch_size=current_batch_size,
                    num_workers=cfg.num_workers,
                )
                print(f'[Schedule] Fold {fold}: switched batch size to {current_batch_size} at epoch {epoch}', flush=True)
            if main_head_frozen_now:
                print(f'[Schedule] Fold {fold}: froze main_head at epoch {epoch}', flush=True)
            if reg_changed and epoch == cfg.reg_boost_epoch:
                print(f'[Schedule] Fold {fold}: boosted weight decay by x{cfg.reg_boost_factor:.2f} at epoch {epoch}', flush=True)

            lrs = get_group_lrs(optimizer)
            bert_encoder_trainable = model.current_unfrozen_bert_layers > 0

            train_stats = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                cfg=cfg,
                bert_encoder_trainable=bert_encoder_trainable,
            )
            val_stats = validate(model=model, loader=val_loader, criterion=criterion, device=device, scaler_t=scaler_t)

            common_wd = next(g['weight_decay'] for g in optimizer.param_groups if g['group_name'] == 'common_fusion')
            xtb_wd = next(g['weight_decay'] for g in optimizer.param_groups if g['group_name'] == 'xtb_encoder')

            history['run_id'].append(run_id)
            history['fold'].append(fold)
            history['epoch'].append(epoch)
            history['stage'].append(stage)
            history['train_loss'].append(train_stats['loss'])
            history['val_loss'].append(val_stats['loss'])
            history['train_mae_std'].append(train_stats['mae_std'])
            history['val_mae_std'].append(val_stats['mae_std'])
            history['val_mae_raw'].append(val_stats['mae_raw'])
            history['best_so_far_val_mae_raw'].append(min(best_mae, float(val_stats['mae_raw'])))
            history['train_z_main_abs'].append(train_stats['z_main_abs'])
            history['val_z_main_abs'].append(val_stats['z_main_abs'])
            history['train_h_xtb_abs'].append(train_stats['h_xtb_abs'])
            history['val_h_xtb_abs'].append(val_stats['h_xtb_abs'])
            history['train_w_xtb_abs'].append(train_stats['w_xtb_abs'])
            history['val_w_xtb_abs'].append(val_stats['w_xtb_abs'])
            history['train_main_pred_abs'].append(train_stats['main_pred_abs'])
            history['val_main_pred_abs'].append(val_stats['main_pred_abs'])
            history['train_bias_pred_abs'].append(train_stats['bias_pred_abs'])
            history['val_bias_pred_abs'].append(val_stats['bias_pred_abs'])
            history['train_pred_abs'].append(train_stats['pred_abs'])
            history['val_pred_abs'].append(val_stats['pred_abs'])
            history['batch_size'].append(current_batch_size)
            history['bert_encoder_lr'].append(lrs['bert_encoder'])
            history['bert_projection_lr'].append(lrs['bert_projection'])
            history['dmpnn_lr'].append(lrs['dmpnn'])
            history['fusion_lr'].append(lrs['common_fusion'])
            history['xtb_lr'].append(lrs['xtb_encoder'])
            history['common_weight_decay'].append(common_wd)
            history['xtb_weight_decay'].append(xtb_wd)

            if val_stats['mae_raw'] < best_mae:
                best_mae = float(val_stats['mae_raw'])
                best_epoch = epoch
                best_stage = stage
                best_lrs = dict(lrs)
                best_common_wd = common_wd
                best_xtb_wd = xtb_wd
                best_val_prediction_frame = build_prediction_frame(
                    run_id=run_id,
                    fold=fold,
                    selected_epoch=epoch,
                    split_manifest_lookup=split_manifest_lookup,
                    val_stats=val_stats,
                )
                best_checkpoint_path = save_best_checkpoint(model, output_dir, fold, epoch, best_mae, cfg, scaler_t, optimizer)
                save_json(
                    {
                        'run_id': run_id,
                        'fold': fold,
                        'best_epoch': best_epoch,
                        'best_val_mae_raw': best_mae,
                        'stage_at_best': stage,
                        'checkpoint_path': best_checkpoint_path,
                        'checkpoint_type': 'full_model',
                        'lrs_at_best': lrs,
                        'weight_decay_at_best': {
                            'common_weight_decay': common_wd,
                            'xtb_weight_decay': xtb_wd,
                        },
                    },
                    os.path.join(output_dir, f'best_fold{fold}_metrics.json'),
                )

            print(
                f'Fold {fold} | Epoch {epoch:03d} | Stage: {stage} | '
                f'Train Loss: {train_stats["loss"]:.5f} | Train MAE(std): {train_stats["mae_std"]:.5f} | '
                f'Val Loss: {val_stats["loss"]:.5f} | Val MAE(std): {val_stats["mae_std"]:.5f} | '
                f'Val MAE(raw): {val_stats["mae_raw"]:.4f} | '
                f'|z_main|(val): {val_stats["z_main_abs"]:.4f} | '
                f'|h_xtb|(val): {val_stats["h_xtb_abs"]:.4f} | '
                f'|w_xtb|(val): {val_stats["w_xtb_abs"]:.4f} | '
                f'|main_pred|(val): {val_stats["main_pred_abs"]:.4f} | '
                f'|b_pred|(val): {val_stats["bias_pred_abs"]:.4f} | '
                f'Best: {best_mae:.4f} (epoch {best_epoch}) | BS: {current_batch_size} | '
                f'LRs: bert_enc={lrs["bert_encoder"]:.2e}, bert_proj={lrs["bert_projection"]:.2e}, '
                f'dmpnn={lrs["dmpnn"]:.2e}, common={lrs["common_fusion"]:.2e}, xtb={lrs["xtb_encoder"]:.2e} | '
                f'WD(common)={common_wd:.2e}, WD(xtb)={xtb_wd:.2e} | noise(z_main)={cfg.main_branch_noise_std:.4f}',
                flush=True,
            )

        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(output_dir, f'fold{fold}_history.csv'), index=False)
        history_df.to_csv(os.path.join(output_dir, f'fold{fold}_epoch_metrics.csv'), index=False)
        plot_fold_curves(history, fold, output_dir)
        all_fold_best_mae.append(best_mae)

        if best_val_prediction_frame is not None:
            save_dataframe(best_val_prediction_frame, os.path.join(output_dir, f'fold{fold}_oof_predictions'), index=False)
            fold_error_bins = build_error_bin_summary(best_val_prediction_frame)
            save_dataframe(fold_error_bins, os.path.join(output_dir, f'fold{fold}_error_bins'), index=False)
            all_oof_frames.append(best_val_prediction_frame)

        fold_record = {
            'run_id': run_id,
            'fold': fold,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'best_epoch': best_epoch,
            'best_stage': best_stage,
            'best_val_mae_raw': best_mae,
            'best_checkpoint_path': best_checkpoint_path,
            'best_common_weight_decay': best_common_wd,
            'best_xtb_weight_decay': best_xtb_wd,
            'best_bert_encoder_lr': best_lrs.get('bert_encoder'),
            'best_bert_projection_lr': best_lrs.get('bert_projection'),
            'best_dmpnn_lr': best_lrs.get('dmpnn'),
            'best_common_lr': best_lrs.get('common_fusion'),
            'best_xtb_lr': best_lrs.get('xtb_encoder'),
        }
        all_fold_records.append(fold_record)
        save_json(fold_record, os.path.join(output_dir, f'fold{fold}_summary.json'))
        print(f'Fold {fold} Best Val MAE(raw): {best_mae:.4f} at epoch {best_epoch}', flush=True)

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    fold_summary_df = pd.DataFrame(all_fold_records)
    if not fold_summary_df.empty:
        save_dataframe(fold_summary_df, os.path.join(output_dir, 'fold_summary'), index=False)

    cv_oof_df = pd.concat(all_oof_frames, ignore_index=True) if all_oof_frames else pd.DataFrame()
    if not cv_oof_df.empty:
        save_dataframe(cv_oof_df, os.path.join(output_dir, 'cv_oof_predictions'), index=False)
        cv_error_bins_df = build_error_bin_summary(cv_oof_df)
        save_dataframe(cv_error_bins_df, os.path.join(output_dir, 'cv_error_bins'), index=False)

    summary = {
        'run_id': run_id,
        'git_commit': git_commit,
        'output_dir': output_dir,
        'checkpoint_policy': 'full_model_only',
        'fold_best_mae_raw': all_fold_best_mae,
        'mean_best_mae_raw': float(np.mean(all_fold_best_mae)),
        'std_best_mae_raw': float(np.std(all_fold_best_mae)),
        'folds_ran': len(selected_folds),
        'folds_requested': selected_folds,
        'total_aligned_samples': total_sample_count,
        'valid_scaffold_samples': valid_scaffold_sample_count,
        'none_scaffold_samples_train_only': len(none_idx),
        'batch_size': cfg.batch_size,
        'stage2_batch_size': cfg.stage2_batch_size,
        'main_branch_noise_std': cfg.main_branch_noise_std,
        'device': str(device),
    }
    save_json(summary, os.path.join(output_dir, 'summary.json'))
    save_dataframe(pd.DataFrame([summary]), os.path.join(output_dir, 'cv_summary'), index=False)
    append_run_summary(
        {
            'run_id': run_id,
            'timestamp': timestamp,
            'git_commit': git_commit,
            'output_dir': output_dir,
            'model_name': cfg.model_name,
            'seed': cfg.seed,
            'n_folds': cfg.n_folds,
            'folds_ran': len(selected_folds),
            'folds_requested': ','.join(map(str, selected_folds)),
            'batch_size': cfg.batch_size,
            'stage2_batch_size': cfg.stage2_batch_size,
            'max_epochs': cfg.max_epochs,
            'freeze_bert_epochs': cfg.freeze_bert_epochs,
            'final_tune_epochs': cfg.final_tune_epochs,
            'bert_encoder_lr': cfg.bert_encoder_lr,
            'bert_projection_lr': cfg.bert_projection_lr,
            'dmpnn_lr': cfg.dmpnn_lr,
            'fusion_lr': cfg.fusion_lr,
            'dropout': cfg.dropout,
            'main_branch_noise_std': cfg.main_branch_noise_std,
                'device': str(device),
            'mean_best_mae_raw': float(np.mean(all_fold_best_mae)),
            'std_best_mae_raw': float(np.std(all_fold_best_mae)),
        },
        cfg.outputs_root,
    )
    plot_summary(all_fold_best_mae, output_dir, selected_folds)

    print('\n========== Clean Residual Boosting Summary ==========', flush=True)
    for fold, mae in zip(selected_folds, all_fold_best_mae):
        print(f'Fold {fold}: Best Val MAE(raw) = {mae:.4f}', flush=True)
    print(f'Mean Best Val MAE(raw): {np.mean(all_fold_best_mae):.4f}', flush=True)
    print(f'Std  Best Val MAE(raw): {np.std(all_fold_best_mae):.4f}', flush=True)
    print(f'\n[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Clean residual boosting training completed.', flush=True)
    print(f'Results saved to: {output_dir}', flush=True)


def main() -> None:
    cfg = parse_args()
    run_training(cfg, infer_device(cfg))


if __name__ == '__main__':
    main()
