from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class RegularizationConfig:
    alpha_dirichlet: float = 0.0
    alpha_connectivity: float = 0.0
    connectivity_eps: float = 1e-2
    connectivity_iters: int = 25
    # Two-sided edge budgeting on expected edge count (prevents over-pruning and densification).
    alpha_edge_budget: float = 0.0
    edge_budget_min_ratio: float = 0.6
    edge_budget_max_ratio: float = 1.4
    # Anchor "stability" penalty (tracks whether anchors were actually enforced).
    alpha_anchor_stability: float = 0.0
    # Primal-dual control knobs.
    dual_lr: float = 1e-2
    cvar_samples: int = 4
    cvar_frac: float = 0.5


@dataclass
class TTAConfig:
    healing_steps: int = 20
    dual_lr: float = 0.01
    primal_lr: float = 1e-2
    kl_beta: float = 1.0
    kl_temp: float = 1.0
    kl_conf: float = 0.0
    cvar_samples: int = 16
    cvar_frac: float = 0.2
    budget_beta: float = 10.0
    budget_hard: bool = False
    budget_dual_scale: float = 5.0
    budget_ref_base: bool = False


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping at the top level.")
    return data


def get_section(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    section = config.get(key, {}) if isinstance(config, dict) else {}
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return section
