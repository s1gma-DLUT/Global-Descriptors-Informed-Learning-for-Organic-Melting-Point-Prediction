#!/usr/bin/env python3
"""
Pragmatic inference entrypoint for plotting and ad-hoc MP predictions.

Modes:
  3d     MolFormer + DMPNN + 17D XTB bundle + 25D RDKit descriptors
  non3d  MolFormer + DMPNN ablation
  both   run both model families and write comparable columns

The model classes are imported from the training scripts on purpose. Keeping
one definition of the architecture is much safer than copying it into an
inference-only script and hoping it never drifts.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, fields
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
for path in (SCRIPT_DIR, REPO_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from legacy_main_train import (  # noqa: E402
    JointCollator as ThreeDCollator,
    JointMolFormerDMPNNXTBModel,
    JointMolGraphTextDataset as ThreeDDataset,
    TrainConfig as ThreeDConfig,
    infer_device as infer_3d_device,
)
from preprocessing import rdkit_features, xtb_extract  # noqa: E402


RDLogger.DisableLog("rdApp.*")


DEFAULT_3D_MODEL_DIR = os.environ.get("MP_3D_MODEL_DIR", "")
DEFAULT_NO3D_MODEL_DIR = os.environ.get("MP_NO3D_MODEL_DIR", "")

RDKIT_25D_FEATURES = [
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "NumValenceElectrons",
    "NumRadicalElectrons",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "LabuteASA",
    "TPSA",
    "MolLogP",
    "MolMR",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "RingCount",
    "NumAliphaticRings",
    "NumAromaticRings",
    "FractionCSP3",
    "qed",
]


@dataclass
class LoadedModel:
    fold: int
    model: torch.nn.Module
    tokenizer: Any
    cfg: Any
    target_scaler: Any
    xtb_imputer: Any = None
    rdkit_imputer: Any = None
    xtb_scaler: Any = None
    rdkit_scaler: Any = None


def load_no3d_module() -> Any:
    module_path = os.path.join(SCRIPT_DIR, "legacy_main_train_mlp_fusion.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"non-3D training script not found: {module_path}")

    spec = importlib.util.spec_from_file_location("legacy_main_train_mlp_fusion_infer", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import non-3D model definitions from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def torch_load(path: str, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def config_from_checkpoint(checkpoint: Dict[str, Any], config_cls: Any) -> Any:
    cfg = config_cls()
    saved = checkpoint.get("config", {})
    valid = {field.name for field in fields(config_cls)}
    for key, value in saved.items():
        if key in valid:
            setattr(cfg, key, value)
    return cfg


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict):
        return {
            key[len("module.") :] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
    return state_dict


def load_state_dict_checked(model: torch.nn.Module, checkpoint: Dict[str, Any], allow_missing_bert: bool) -> None:
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:30]}")

    if missing:
        allowed_prefixes = ("bert_encoder.encoder.",) if allow_missing_bert else tuple()
        required_missing = [key for key in missing if not key.startswith(allowed_prefixes)]
        if required_missing:
            raise RuntimeError(f"Missing required checkpoint keys: {required_missing[:30]}")
        print(f"[INFO] Missing {len(missing)} BERT encoder keys; using pretrained initialization.", flush=True)


def require_file(path: str, label: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_joblib(path: str, label: str) -> Any:
    return joblib.load(require_file(path, label))


def load_3d_fold(
    model_dir: str,
    fold: int,
    device: torch.device,
    model_name_override: str = "",
    allow_missing_bert: bool = False,
) -> LoadedModel:
    ckpt_path = require_file(
        os.path.join(model_dir, f"best_fold{fold}_full_model.pt"),
        f"3D fold {fold} checkpoint",
    )
    checkpoint = torch_load(ckpt_path, device)
    cfg = config_from_checkpoint(checkpoint, ThreeDConfig)
    if model_name_override:
        cfg.model_name = model_name_override

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    model = JointMolFormerDMPNNXTBModel(cfg=cfg, xtb_dim=17, rdkit_dim=25).to(device)
    load_state_dict_checked(model, checkpoint, allow_missing_bert=allow_missing_bert)
    model.eval()

    scaler_dir = require_file(os.path.join(model_dir, f"scalers_fold{fold}"), f"3D fold {fold} scaler dir")
    return LoadedModel(
        fold=fold,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        target_scaler=load_joblib(os.path.join(scaler_dir, "target_scaler.pkl"), "target scaler"),
        xtb_imputer=load_joblib(os.path.join(scaler_dir, "xtb_imputer.pkl"), "XTB imputer"),
        rdkit_imputer=load_joblib(os.path.join(scaler_dir, "rdkit_imputer.pkl"), "RDKit imputer"),
        xtb_scaler=load_joblib(os.path.join(scaler_dir, "xtb_scaler.pkl"), "XTB scaler"),
        rdkit_scaler=load_joblib(os.path.join(scaler_dir, "rdkit_scaler.pkl"), "RDKit scaler"),
    )


def load_no3d_fold(
    no3d_module: Any,
    model_dir: str,
    fold: int,
    device: torch.device,
    model_name_override: str = "",
    allow_missing_bert: bool = False,
) -> LoadedModel:
    ckpt_path = require_file(
        os.path.join(model_dir, f"best_fold{fold}_full_model.pt"),
        f"non-3D fold {fold} checkpoint",
    )
    checkpoint = torch_load(ckpt_path, device)
    cfg = config_from_checkpoint(checkpoint, no3d_module.TrainConfig)
    if model_name_override:
        cfg.model_name = model_name_override

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    model = no3d_module.JointMolFormerDMPNNMainOnlyModel(cfg=cfg).to(device)
    load_state_dict_checked(model, checkpoint, allow_missing_bert=allow_missing_bert)
    model.eval()

    scaler_dir = require_file(os.path.join(model_dir, f"scalers_fold{fold}"), f"non-3D fold {fold} scaler dir")
    return LoadedModel(
        fold=fold,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        target_scaler=load_joblib(os.path.join(scaler_dir, "target_scaler.pkl"), "target scaler"),
    )


def validate_smiles(smiles: Any) -> str:
    if pd.isna(smiles):
        raise ValueError("SMILES is missing")
    smiles_str = str(smiles).strip()
    if not smiles_str:
        raise ValueError("SMILES is empty")
    if Chem.MolFromSmiles(smiles_str) is None:
        raise ValueError(f"Invalid SMILES: {smiles_str}")
    return smiles_str


def rdkit25_from_smiles(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    values: List[float] = []
    for name in RDKIT_25D_FEATURES:
        func = getattr(Descriptors, name, None)
        if func is None:
            raise RuntimeError(f"RDKit descriptor is unavailable: {name}")
        value = float(func(mol))
        if not np.isfinite(value):
            raise RuntimeError(f"RDKit descriptor {name} returned {value} for {smiles}")
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def write_xyz_from_smiles(smiles: str, xyz_path: str, seed: int = 61453) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    status = AllChem.EmbedMolecule(mol, params)
    if status == -1:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
    if status == -1:
        raise RuntimeError(f"RDKit 3D embedding failed for {smiles}")

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            opt_status = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            opt_status = AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception as exc:
        raise RuntimeError(f"RDKit geometry optimization failed for {smiles}: {exc}") from exc
    if opt_status == -1:
        raise RuntimeError(f"RDKit geometry optimization failed for {smiles}")

    conf = mol.GetConformer()
    with open(xyz_path, "w", encoding="utf-8") as handle:
        handle.write(f"{mol.GetNumAtoms()}\n")
        handle.write(f"Generated from SMILES: {smiles}\n")
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            handle.write(f"{atom.GetSymbol()} {pos.x:.8f} {pos.y:.8f} {pos.z:.8f}\n")


def read_xtb_texts(work_dir: str, stdout: str, stderr: str) -> str:
    chunks = [stdout or "", stderr or ""]
    patterns = ["*.log", "*.out", "xtbopt.log", "xtb.log"]
    seen: set[str] = set()
    for pattern in patterns:
        for path in glob.glob(os.path.join(work_dir, pattern)):
            if path in seen or not os.path.isfile(path):
                continue
            seen.add(path)
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                chunks.append(handle.read())
    return "\n".join(chunks)


def xtb17_from_smiles(
    smiles: str,
    xtb_cmd: Sequence[str],
    tmp_root: str,
    timeout: int,
    keep_xtb_dirs: bool = False,
) -> np.ndarray:
    work_dir = tempfile.mkdtemp(prefix="mp_xtb_", dir=tmp_root)
    xyz_path = os.path.join(work_dir, "mol.xyz")
    try:
        write_xyz_from_smiles(smiles, xyz_path)
        cmd = list(xtb_cmd) + [xyz_path, "--gfn2", "--opt"]
        proc = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        xtb_text = read_xtb_texts(work_dir, proc.stdout, proc.stderr)
        if proc.returncode != 0:
            tail = "\n".join(xtb_text.splitlines()[-25:])
            raise RuntimeError(f"XTB failed for {smiles} with code {proc.returncode}:\n{tail}")

        parsed = xtb_extract.parse_xtb_output(xtb_text, smiles)
        if not parsed.success:
            notes = "; ".join(parsed.raw_parsing_notes[-5:])
            raise RuntimeError(f"XTB parse failed for {smiles}: {parsed.error_message or notes}")
        xtb16 = xtb_extract.xtb_result_to_feature_vector(parsed)
        if xtb16.shape != (16,) or not np.isfinite(xtb16).all():
            raise RuntimeError(f"Bad XTB 16D vector for {smiles}: shape={xtb16.shape}")

        volume = rdkit_features.compute_molecular_volume_cm3_mol(smiles)
        if volume is None or not np.isfinite(float(volume)):
            raise RuntimeError(f"RDKit molecular volume failed for {smiles}")
        return np.concatenate([xtb16, np.asarray([volume], dtype=np.float32)]).astype(np.float32)
    finally:
        if keep_xtb_dirs:
            print(f"[INFO] Kept XTB work dir: {work_dir}", flush=True)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


def build_3d_features(
    smiles_list: Sequence[str],
    xtb_cmd: Sequence[str],
    tmp_root: str,
    timeout: int,
    keep_xtb_dirs: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    xtb_rows: List[np.ndarray] = []
    rdkit_rows: List[np.ndarray] = []
    cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    total = len(smiles_list)
    for idx, smiles in enumerate(smiles_list, start=1):
        if smiles not in cache:
            print(f"[3D features] {idx}/{total}: {smiles}", flush=True)
            cache[smiles] = (
                xtb17_from_smiles(smiles, xtb_cmd, tmp_root, timeout, keep_xtb_dirs),
                rdkit25_from_smiles(smiles),
            )
        xtb_row, rdkit_row = cache[smiles]
        xtb_rows.append(xtb_row)
        rdkit_rows.append(rdkit_row)
    return np.vstack(xtb_rows).astype(np.float32), np.vstack(rdkit_rows).astype(np.float32)


def make_loader_3d(
    loaded: LoadedModel,
    smiles: Sequence[str],
    xtb_raw: np.ndarray,
    rdkit_raw: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    xtb_scaled = loaded.xtb_scaler.transform(loaded.xtb_imputer.transform(xtb_raw))
    rdkit_scaled = loaded.rdkit_scaler.transform(loaded.rdkit_imputer.transform(rdkit_raw))
    dataset = ThreeDDataset(
        smiles_list=smiles,
        xtb_features=xtb_scaled,
        rdkit_features=rdkit_scaled,
        targets=np.zeros(len(smiles), dtype=np.float32),
        sample_ids=np.arange(len(smiles)),
        cache_graphs=bool(getattr(loaded.cfg, "cache_graphs", True)),
    )
    collator = ThreeDCollator(loaded.tokenizer, max_length=int(getattr(loaded.cfg, "max_length", 200)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )


def make_loader_no3d(
    no3d_module: Any,
    loaded: LoadedModel,
    smiles: Sequence[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = no3d_module.JointMolGraphTextDataset(
        smiles_list=smiles,
        targets=np.zeros(len(smiles), dtype=np.float32),
        sample_ids=np.arange(len(smiles)),
        cache_graphs=bool(getattr(loaded.cfg, "cache_graphs", True)),
    )
    collator = no3d_module.JointCollator(loaded.tokenizer, max_length=int(getattr(loaded.cfg, "max_length", 200)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )


def predict_loaded_3d(
    loaded: LoadedModel,
    smiles: Sequence[str],
    xtb_raw: np.ndarray,
    rdkit_raw: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    loader = make_loader_3d(loaded, smiles, xtb_raw, rdkit_raw, batch_size, num_workers, device)
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            pred = loaded.model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                graph_batch=batch["graph_batch"].to(device),
                xtb_feat=batch["xtb"].to(device),
                rdkit_feat=batch["rdkit"].to(device),
            )
            chunks.append(pred.detach().cpu().numpy())
    pred_std = np.concatenate(chunks, axis=0)
    return loaded.target_scaler.inverse_transform(pred_std.reshape(-1, 1)).reshape(-1)


def predict_loaded_no3d(
    no3d_module: Any,
    loaded: LoadedModel,
    smiles: Sequence[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    loader = make_loader_no3d(no3d_module, loaded, smiles, batch_size, num_workers, device)
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            pred = loaded.model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                graph_batch=batch["graph_batch"].to(device),
            )
            chunks.append(pred.detach().cpu().numpy())
    pred_std = np.concatenate(chunks, axis=0)
    return loaded.target_scaler.inverse_transform(pred_std.reshape(-1, 1)).reshape(-1)


def parse_folds(folds: str, fold: Optional[int]) -> List[int]:
    if folds.strip():
        values = [int(part.strip()) for part in folds.split(",") if part.strip()]
    elif fold is not None:
        values = [fold]
    else:
        values = [1]
    if not values:
        raise ValueError("No folds selected")
    return values


def resolve_smiles_col(df: pd.DataFrame, requested: str) -> str:
    candidates = [requested, "SMILES", "smiles"]
    for col in candidates:
        if col and col in df.columns:
            return col
    raise ValueError(f"SMILES column not found. Tried: {candidates}. Columns: {df.columns.tolist()}")


def append_fold_predictions(result: pd.DataFrame, indices: Sequence[int], prefix: str, fold_preds: Dict[int, np.ndarray]) -> None:
    if not fold_preds:
        return
    matrix = np.vstack([fold_preds[fold] for fold in sorted(fold_preds)])
    for fold, preds in fold_preds.items():
        result.loc[list(indices), f"{prefix}_fold{fold}"] = preds
    result.loc[list(indices), f"{prefix}_mean"] = matrix.mean(axis=0)
    result.loc[list(indices), f"{prefix}_std"] = matrix.std(axis=0)


def load_models(args: argparse.Namespace, device: torch.device, folds: Sequence[int]) -> Tuple[List[LoadedModel], Any, List[LoadedModel]]:
    models_3d: List[LoadedModel] = []
    models_no3d: List[LoadedModel] = []
    no3d_module = None

    if args.mode in ("3d", "both"):
        for fold in folds:
            print(f"[load] 3D fold {fold}", flush=True)
            models_3d.append(
                load_3d_fold(
                    args.model_dir_3d,
                    fold,
                    device,
                    model_name_override=args.model_name,
                    allow_missing_bert=args.allow_missing_bert,
                )
            )

    if args.mode in ("non3d", "both"):
        no3d_module = load_no3d_module()
        for fold in folds:
            print(f"[load] non-3D fold {fold}", flush=True)
            models_no3d.append(
                load_no3d_fold(
                    no3d_module,
                    args.model_dir_no3d,
                    fold,
                    device,
                    model_name_override=args.model_name,
                    allow_missing_bert=args.allow_missing_bert,
                )
            )

    return models_3d, no3d_module, models_no3d


def predict_single(args: argparse.Namespace, device: torch.device, models_3d: List[LoadedModel], no3d_module: Any, models_no3d: List[LoadedModel]) -> None:
    smiles = validate_smiles(args.smiles)
    print(f"SMILES: {smiles}", flush=True)

    if models_3d:
        with tempfile.TemporaryDirectory(prefix="mp_predict_", dir=args.xtb_work_dir) as tmp_root:
            xtb_raw, rdkit_raw = build_3d_features(
                [smiles],
                args.xtb_cmd,
                tmp_root,
                args.xtb_timeout,
                args.keep_xtb_dirs,
            )
            fold_preds = {
                loaded.fold: predict_loaded_3d(
                    loaded,
                    [smiles],
                    xtb_raw,
                    rdkit_raw,
                    args.batch_size,
                    args.num_workers,
                    device,
                )[0]
                for loaded in models_3d
            }
        values = np.asarray(list(fold_preds.values()), dtype=float)
        print(f"3D prediction: {values.mean():.2f} degC (std {values.std():.2f})", flush=True)
        for fold, value in fold_preds.items():
            print(f"  fold{fold}: {value:.2f}", flush=True)

    if models_no3d:
        fold_preds = {
            loaded.fold: predict_loaded_no3d(
                no3d_module,
                loaded,
                [smiles],
                args.batch_size,
                args.num_workers,
                device,
            )[0]
            for loaded in models_no3d
        }
        values = np.asarray(list(fold_preds.values()), dtype=float)
        print(f"non-3D prediction: {values.mean():.2f} degC (std {values.std():.2f})", flush=True)
        for fold, value in fold_preds.items():
            print(f"  fold{fold}: {value:.2f}", flush=True)


def predict_csv(args: argparse.Namespace, device: torch.device, models_3d: List[LoadedModel], no3d_module: Any, models_no3d: List[LoadedModel]) -> None:
    df = pd.read_csv(args.input_csv)
    smiles_col = resolve_smiles_col(df, args.smiles_col)
    result = df.copy()
    result["predict_error"] = ""

    valid_indices: List[int] = []
    valid_smiles: List[str] = []
    for idx, value in df[smiles_col].items():
        try:
            valid_smiles.append(validate_smiles(value))
            valid_indices.append(int(idx))
        except Exception as exc:
            if args.fail_fast:
                raise
            result.at[idx, "predict_error"] = str(exc)

    if not valid_smiles:
        raise ValueError("No valid SMILES rows to predict")

    if models_no3d:
        print(f"[predict] non-3D rows: {len(valid_smiles)}", flush=True)
        fold_preds = {
            loaded.fold: predict_loaded_no3d(
                no3d_module,
                loaded,
                valid_smiles,
                args.batch_size,
                args.num_workers,
                device,
            )
            for loaded in models_no3d
        }
        append_fold_predictions(result, valid_indices, "pred_non3d", fold_preds)

    if models_3d:
        xtb_indices: List[int] = []
        xtb_smiles: List[str] = []
        xtb_raw_rows: List[np.ndarray] = []
        rdkit_raw_rows: List[np.ndarray] = []
        feature_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        with tempfile.TemporaryDirectory(prefix="mp_predict_", dir=args.xtb_work_dir) as tmp_root:
            total = len(valid_smiles)
            for pos, (row_idx, smiles) in enumerate(zip(valid_indices, valid_smiles), start=1):
                try:
                    if smiles not in feature_cache:
                        print(f"[3D features] {pos}/{total}: {smiles}", flush=True)
                        feature_cache[smiles] = (
                            xtb17_from_smiles(
                                smiles,
                                args.xtb_cmd,
                                tmp_root,
                                args.xtb_timeout,
                                args.keep_xtb_dirs,
                            ),
                            rdkit25_from_smiles(smiles),
                        )
                    xtb_row, rdkit_row = feature_cache[smiles]
                    xtb_indices.append(row_idx)
                    xtb_smiles.append(smiles)
                    xtb_raw_rows.append(xtb_row)
                    rdkit_raw_rows.append(rdkit_row)
                except Exception as exc:
                    if args.fail_fast:
                        raise
                    old = str(result.at[row_idx, "predict_error"])
                    result.at[row_idx, "predict_error"] = (old + "; " if old else "") + f"3D failed: {exc}"

        if xtb_smiles:
            xtb_raw = np.vstack(xtb_raw_rows).astype(np.float32)
            rdkit_raw = np.vstack(rdkit_raw_rows).astype(np.float32)
            print(f"[predict] 3D rows: {len(xtb_smiles)}", flush=True)
            fold_preds = {
                loaded.fold: predict_loaded_3d(
                    loaded,
                    xtb_smiles,
                    xtb_raw,
                    rdkit_raw,
                    args.batch_size,
                    args.num_workers,
                    device,
                )
                for loaded in models_3d
            }
            append_fold_predictions(result, xtb_indices, "pred_3d", fold_preds)
        elif args.fail_fast:
            raise ValueError("No rows could produce 3D features")

    if "pred_3d_mean" in result.columns and "pred_non3d_mean" in result.columns:
        result["pred_delta_3d_minus_non3d"] = result["pred_3d_mean"] - result["pred_non3d_mean"]

    output_csv = args.output_csv or "predictions.csv"
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Predictions saved to: {output_csv}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MP inference for 3D/non-3D fold checkpoints.")
    parser.add_argument("--mode", choices=["3d", "non3d", "both"], default="both")
    parser.add_argument("--smiles", default="", help="Single SMILES to predict.")
    parser.add_argument("--input_csv", default="", help="CSV with a SMILES/SMILES column.")
    parser.add_argument("--smiles_col", default="smiles", help="SMILES column; falls back to SMILES/smiles.")
    parser.add_argument("--output_csv", default="", help="Output CSV path for batch prediction.")
    parser.add_argument("--model_dir_3d", default=DEFAULT_3D_MODEL_DIR)
    parser.add_argument("--model_dir_no3d", default=DEFAULT_NO3D_MODEL_DIR)
    parser.add_argument("--model_name", default="", help="Override model_name stored in checkpoints.")
    parser.add_argument("--device", default="", help="cuda:0, cuda:1, cpu, etc. Empty means training default.")
    parser.add_argument("--fold", type=int, default=1, help="Single fold. Ignored when --folds is set.")
    parser.add_argument("--folds", default="", help="Comma-separated folds, e.g. 1,3,5.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--xtb_work_dir", default=None, help="Parent temp dir for XTB jobs.")
    parser.add_argument("--xtb_timeout", type=int, default=900)
    parser.add_argument(
        "--xtb_cmd",
        nargs="+",
        default=shlex.split(os.environ.get("XTB_CMD", "conda run -n xtb xtb")),
        help="Command prefix for XTB. Example: --xtb_cmd xtb",
    )
    parser.add_argument("--allow_missing_bert", action="store_true", help="Allow checkpoints without BERT encoder weights.")
    parser.add_argument("--fail_fast", action="store_true", help="Abort CSV prediction on the first bad row.")
    parser.add_argument("--keep_xtb_dirs", action="store_true", help="Keep temporary XTB directories for debugging.")
    args = parser.parse_args()

    if bool(args.smiles) == bool(args.input_csv):
        parser.error("Provide exactly one of --smiles or --input_csv.")
    if args.mode in ("3d", "both") and not args.model_dir_3d:
        parser.error("Provide --model_dir_3d or set MP_3D_MODEL_DIR.")
    if args.mode in ("non3d", "both") and not args.model_dir_no3d:
        parser.error("Provide --model_dir_no3d or set MP_NO3D_MODEL_DIR.")
    if args.xtb_work_dir is None:
        args.xtb_work_dir = tempfile.gettempdir()
    if isinstance(args.xtb_cmd, str):
        args.xtb_cmd = shlex.split(args.xtb_cmd)
    elif len(args.xtb_cmd) == 1:
        args.xtb_cmd = shlex.split(args.xtb_cmd[0])
    return args


def main() -> None:
    args = parse_args()
    folds = parse_folds(args.folds, args.fold)

    cfg = ThreeDConfig()
    if args.device:
        cfg.device = args.device
    device = infer_3d_device(cfg)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; using CPU.", flush=True)
        device = torch.device("cpu")

    print(f"Device: {device} | mode: {args.mode} | folds: {folds}", flush=True)
    print(f"XTB command: {' '.join(args.xtb_cmd)}", flush=True)

    models_3d, no3d_module, models_no3d = load_models(args, device, folds)
    if args.smiles:
        predict_single(args, device, models_3d, no3d_module, models_no3d)
    else:
        predict_csv(args, device, models_3d, no3d_module, models_no3d)


if __name__ == "__main__":
    main()
