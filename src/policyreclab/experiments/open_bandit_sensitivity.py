from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from policyreclab.datasets import load_open_bandit_csv

@dataclass(frozen=True)
class OpenBanditSensitivityResult:
    campaign: str
    on_policy_reference_ctr: float
    ips_value: float
    snips_value: float
    clipped_ips_values: dict[float, float]
    max_weight: float
    p99_weight: float
    p999_weight: float
    effective_sample_size: float
    effective_sample_fraction: float
    top_1pct_weight_share: float
    top_01pct_weight_share: float
    top_001pct_weight_share: float
    top_1pct_weighted_reward_share: float

def _top_share(values: np.ndarray, fraction: float) -> float:
    total=float(np.sum(values))
    if total == 0.0: return 0.0
    k=max(1, int(np.ceil(values.size*fraction)))
    top=np.partition(values, values.size-k)[values.size-k:]
    return float(np.sum(top)/total)

def run_bts_to_random_sensitivity(*, bts_csv: str|Path, random_csv: str|Path,
    campaign: str, n_actions: int|None=None,
    clip_thresholds: tuple[float,...]=(10.0,20.0,50.0,100.0)
) -> OpenBanditSensitivityResult:
    bts=load_open_bandit_csv(bts_csv,n_actions=n_actions)
    random=load_open_bandit_csv(random_csv,n_actions=n_actions)
    k=n_actions or max(bts.n_actions,random.n_actions)
    if any(c<=0 for c in clip_thresholds): raise ValueError("clip thresholds must be positive")
    w=(1.0/k)/bts.behavior_propensities
    wr=w*bts.rewards
    sw=float(np.sum(w))
    ess=float(sw**2/np.sum(w**2))
    return OpenBanditSensitivityResult(
        campaign=campaign, on_policy_reference_ctr=random.empirical_ctr,
        ips_value=float(np.mean(wr)), snips_value=float(np.sum(wr)/sw),
        clipped_ips_values={float(c):float(np.mean(np.minimum(w,c)*bts.rewards)) for c in clip_thresholds},
        max_weight=float(np.max(w)), p99_weight=float(np.quantile(w,.99)),
        p999_weight=float(np.quantile(w,.999)), effective_sample_size=ess,
        effective_sample_fraction=ess/bts.n_rounds,
        top_1pct_weight_share=_top_share(w,.01), top_01pct_weight_share=_top_share(w,.001),
        top_001pct_weight_share=_top_share(w,.0001),
        top_1pct_weighted_reward_share=_top_share(wr,.01))
