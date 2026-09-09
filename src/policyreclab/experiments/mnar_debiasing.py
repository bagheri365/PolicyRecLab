from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from policyreclab.datasets.mnar_ratings import RatingTriples

@dataclass(frozen=True)
class MNARDebiasingResult:
    dataset: str
    n_observational: int
    n_randomized: int
    naive_observational_mean: float
    randomized_reference_mean: float
    naive_bias: float
    ips_observational_mean: float|None
    ips_bias: float|None
    propensity_source: str
    interpretation: str

def inverse_item_frequency_propensities(
    observational: RatingTriples, *, n_users: int, n_items: int
) -> np.ndarray:
    """Simple exposure model P(O_ui=1 | item), estimated from observational data."""
    counts=np.bincount(observational.items,minlength=n_items).astype(float)
    p=counts/float(n_users)
    return np.clip(p,1e-12,1.0)

def run_mnar_mean_debiasing(
    *, dataset: str, observational: RatingTriples, randomized: RatingTriples,
    n_users: int, n_items: int, item_propensities: np.ndarray|None=None,
    propensity_source: str="estimated_item_frequency"
) -> MNARDebiasingResult:
    """Compare observational rating mean with a randomized reference.

    This is an MNAR missingness/debiasing benchmark, not contextual-bandit OPE.
    """
    naive=float(np.mean(observational.ratings))
    reference=float(np.mean(randomized.ratings))
    if item_propensities is None:
        item_propensities=inverse_item_frequency_propensities(
            observational,n_users=n_users,n_items=n_items)
    p=np.asarray(item_propensities,dtype=float)
    if p.shape!=(n_items,): raise ValueError("item_propensities must have shape (n_items,)")
    if np.any((p<=0)|(p>1)): raise ValueError("item propensities must lie in (0,1]")
    w=1.0/p[observational.items]
    ips=float(np.sum(w*observational.ratings)/np.sum(w))
    return MNARDebiasingResult(
        dataset=dataset,n_observational=observational.n_ratings,
        n_randomized=randomized.n_ratings,naive_observational_mean=naive,
        randomized_reference_mean=reference,naive_bias=naive-reference,
        ips_observational_mean=ips,ips_bias=ips-reference,
        propensity_source=propensity_source,
        interpretation="randomized ratings are an empirical reference; this is MNAR debiasing, not production contextual-bandit OPE")
