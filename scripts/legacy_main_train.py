import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse
import random
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
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
    model_name: str = 'PATH_OR_HF_ID_TO_MOLFORMER'
    split_dir: str = 'splits/scaffold'
    use_random_split: bool = False  # If True, use random K-fold instead of frozen scaffold split
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


def parse_args_to_config() -> TrainConfig:
    """Parse command-line overrides for TrainConfig fields."""
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description='Train multimodal melting point model')
    parser.add_argument('--use_frozen_split', action='store_true', help='Use frozen scaffold splits')

    field_map = {field.name: field for field in fields(TrainConfig)}
    for name, field in field_map.items():
        if name == 'use_random_split':
            parser.add_argument(f'--{name}', action='store_true')
        elif isinstance(getattr(cfg, name), bool):
            parser.add_argument(f'--{name}', type=lambda x: str(x).lower() in {'1', 'true', 'yes'})
        else:
            parser.add_argument(f'--{name}', type=type(getattr(cfg, name)))

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f'[WARN] Ignoring unsupported arguments: {unknown}', flush=True)

    for name in field_map:
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)

    if args.use_frozen_split:
        cfg.use_random_split = False

    if 'PATH_OR_HF_ID_TO_MOLFORMER' in cfg.model_name:
        cfg.model_name = os.environ.get('MOLFORMER_MODEL', cfg.model_name)
    if 'PATH_OR_HF_ID_TO_MOLFORMER' in cfg.model_name:
        raise ValueError(
            'Set --model_name, set model_name_or_path in the YAML config, '
            'or export MOLFORMER_MODEL.'
        )
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_run_basename(timestamp: str, cfg: TrainConfig) -> str:
    """Generate run basename from timestamp and config."""
    split_name = 'random' if cfg.use_random_split else 'frozen_scaffold'
    parts = [
        'mp',
        split_name,
        f'fold{cfg.n_folds}',
        f'bs{cfg.batch_size}',
        f'seed{cfg.seed}',
        timestamp
    ]
    if cfg.output_tag:
        parts.insert(1, cfg.output_tag)
    return '_'.join(parts)


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


def load_frozen_fold_indices(split_dir: str, fold: int, total_samples: int) -> Tuple[List[int], List[int]]:
    val_file = os.path.join(split_dir, f'fold_{fold}_val_idx.npy')
    if not os.path.exists(val_file):
        raise FileNotFoundError(f'Frozen validation index file not found: {val_file}')
    val_idx = sorted(np.load(val_file).astype(int).tolist())
    if any(idx < 0 or idx >= total_samples for idx in val_idx):
        raise ValueError(f'Validation indices for fold {fold} are out of range.')
    val_idx_set = set(val_idx)
    train_idx = [idx for idx in range(total_samples) if idx not in val_idx_set]
    return train_idx, val_idx


def build_random_folds(n_samples: int, n_folds: int = 5, seed: int = 114514) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_sizes = np.full(n_folds, n_samples // n_folds)
    fold_sizes[:n_samples % n_folds] += 1

    fold_indices: List[List[int]] = []
    start = 0
    for fold_size in fold_sizes:
        fold_indices.append(indices[start:start + fold_size].tolist())
        start += fold_size

    return fold_indices


def get_random_fold_indices(fold_indices: List[List[int]], fold: int, total_samples: int) -> Tuple[List[int], List[int]]:
    fold_zero = fold - 1
    val_idx = sorted(fold_indices[fold_zero])
    val_idx_set = set(val_idx)
    all_indices = set(range(total_samples))
    train_idx = sorted(list(all_indices - val_idx_set))

    return train_idx, val_idx


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


def load_aligned_multimodal_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Loading joint data from {data_dir}...', flush=True)
    train_csv = first_existing_path([
        os.path.join(data_dir, 'raw', 'multimodal_train.csv'),
        os.path.join(data_dir, 'raw', 'cleaned', 'data_set.csv'),
    ])
    rdkit_path = first_existing_path([
        os.path.join(data_dir, 'processed', 'rdkit3d_train.npy'),
        os.path.join(data_dir, 'raw', 'cleaned', 'rdkit3d_train.npy'),
    ])
    xtb_path = first_existing_path([
        os.path.join(data_dir, 'processed', 'XTB_train.pth'),
        os.path.join(data_dir, 'raw', 'cleaned', 'XTB_train.pth'),
    ])

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


def first_existing_path(paths: Sequence[str]) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'None of these paths exists: {paths}')


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
    return full_path


def run_training(cfg: TrainConfig, device: torch.device) -> None:
    set_seed(cfg.seed)
    selected_folds = parse_selected_folds(cfg)
    if not selected_folds:
        raise ValueError('No folds selected to run.')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_basename = get_run_basename(timestamp, cfg)
    output_dir = os.path.join(cfg.outputs_root, run_basename)
    os.makedirs(output_dir, exist_ok=True)
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Clean residual boosting training script started', flush=True)
    print(f'Output dir: {output_dir}', flush=True)
    print(f'Using device: {device}', flush=True)

    targets_raw, xtb_raw, rdkit_raw, smiles_list = load_aligned_multimodal_data(cfg.data_dir)
    total_sample_count = len(smiles_list)

    if cfg.use_random_split:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Using random K-fold split with seed {cfg.seed}', flush=True)
        fold_indices_list = build_random_folds(total_sample_count, n_folds=cfg.n_folds, seed=cfg.seed)
        none_idx = []
    else:
        split_dir = os.path.abspath(cfg.split_dir)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Using frozen scaffold split from {split_dir}', flush=True)
        none_file = os.path.join(split_dir, 'none_idx.npy')
        none_idx = np.load(none_file).astype(int).tolist() if os.path.exists(none_file) else []

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    collator = JointCollator(tokenizer=tokenizer, max_length=cfg.max_length)

    none_idx_set = set(none_idx)
    all_fold_best_mae: List[float] = []

    for fold in selected_folds:
        if cfg.use_random_split:
            train_idx, val_idx = get_random_fold_indices(fold_indices_list, fold, total_sample_count)
            split_type_str = 'random_kfold'
            none_count_str = ''
        else:
            train_idx, val_idx = load_frozen_fold_indices(split_dir, fold, total_sample_count)
            split_type_str = 'frozen_scaffold'
            none_count_str = f' | None(train-only): {len(none_idx)}'

        val_idx_set = set(val_idx)

        if not cfg.use_random_split:
            assert not (none_idx_set & val_idx_set), 'None-scaffold samples leaked into validation set.'
            assert none_idx_set.issubset(set(train_idx)), 'Some none-scaffold samples are missing from the frozen train split.'

        assert len(set(train_idx) & val_idx_set) == 0, 'Train/Val overlap detected.'

        print('\n' + '=' * 80, flush=True)
        print(
            f'Fold {fold}/{cfg.n_folds} | Split: {split_type_str} | '
            f'Train: {len(train_idx)} | Val: {len(val_idx)}{none_count_str}',
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
        best_checkpoint_path = ''

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

            if val_stats['mae_raw'] < best_mae:
                best_mae = float(val_stats['mae_raw'])
                best_epoch = epoch
                best_checkpoint_path = save_best_checkpoint(model, output_dir, fold, epoch, best_mae, cfg, scaler_t, optimizer)

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

        all_fold_best_mae.append(best_mae)

        print(
            f'Fold {fold} Best Val MAE(raw): {best_mae:.4f} at epoch {best_epoch} | '
            f'checkpoint: {best_checkpoint_path}',
            flush=True,
        )

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print('\n========== Clean Residual Boosting Summary ==========', flush=True)
    for fold, mae in zip(selected_folds, all_fold_best_mae):
        print(f'Fold {fold}: Best Val MAE(raw) = {mae:.4f}', flush=True)
    print(f'Mean Best Val MAE(raw): {np.mean(all_fold_best_mae):.4f}', flush=True)
    print(f'Std  Best Val MAE(raw): {np.std(all_fold_best_mae):.4f}', flush=True)
    print(f'\n[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Clean residual boosting training completed.', flush=True)
    print(f'Results saved to: {output_dir}', flush=True)


def main() -> None:
    cfg = parse_args_to_config()
    run_training(cfg, infer_device(cfg))


if __name__ == '__main__':
    main()
