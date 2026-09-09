from __future__ import annotations

import argparse
from pathlib import Path

from policyreclab.experiments import run_bts_to_random_replication


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the Uniform Random policy from Open Bandit Dataset BTS "
            "logs and compare it with Random's separate empirical CTR."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--campaign", choices=("all", "men", "women"), default="all")
    parser.add_argument("--n-actions", type=int, default=None)
    args = parser.parse_args()

    result = run_bts_to_random_replication(
        bts_csv=args.data_root / "bts" / args.campaign / f"{args.campaign}.csv",
        random_csv=(
            args.data_root / "random" / args.campaign / f"{args.campaign}.csv"
        ),
        campaign=args.campaign,
        n_actions=args.n_actions,
    )

    print(f"campaign: {result.campaign}")
    print(f"behavior policy: {result.behavior_policy}")
    print(f"target policy: {result.target_policy}")
    print(f"behavior rounds: {result.n_behavior_rounds}")
    print(f"reference rounds: {result.n_reference_rounds}")
    print(f"n_actions: {result.n_actions}")
    print(f"IPS estimate: {result.ips_policy_value:.8f}")
    print(f"Random on-policy empirical CTR: {result.on_policy_reference_ctr:.8f}")
    print(f"signed error: {result.absolute_error:.8f}")
    print(f"relative error: {result.relative_error:.4%}")
    print(f"max weight: {result.weight_diagnostics.maximum:.4f}")
    print(f"p99 weight: {result.weight_diagnostics.p99:.4f}")
    print(
        "ESS heuristic: "
        f"{result.weight_diagnostics.effective_sample_size:.1f} "
        f"({result.weight_diagnostics.effective_sample_fraction:.2%})"
    )


if __name__ == "__main__":
    main()
