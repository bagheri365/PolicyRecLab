from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from policyreclab.experiments.open_bandit_sensitivity import run_bts_to_random_sensitivity

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--campaigns",nargs="+",default=("all","men","women"),choices=("all","men","women"))
    p.add_argument("--clip-thresholds",nargs="+",type=float,default=(10.,20.,50.,100.))
    p.add_argument("--artifact-dir",type=Path,default=Path("artifacts/v1_2_open_bandit_dataset"))
    a=p.parse_args(); results=[]
    for c in a.campaigns:
        results.append(run_bts_to_random_sensitivity(
            bts_csv=a.data_root/"bts"/c/f"{c}.csv",random_csv=a.data_root/"random"/c/f"{c}.csv",
            campaign=c,clip_thresholds=tuple(a.clip_thresholds)))
    a.artifact_dir.mkdir(parents=True,exist_ok=True)
    fields=["campaign","reference_ctr","ips","ips_relative_error","snips","snips_relative_error",
            "max_weight","p99_weight","p999_weight","ess","ess_fraction",
            "top_1pct_weight_share","top_01pct_weight_share","top_001pct_weight_share",
            "top_1pct_weighted_reward_share"]
    for c in a.clip_thresholds: fields += [f"clipped_ips_{c:g}",f"clipped_ips_{c:g}_relative_error"]
    with (a.artifact_dir/"summary.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for r in results:
            ref=r.on_policy_reference_ctr
            row={"campaign":r.campaign,"reference_ctr":ref,"ips":r.ips_value,
                 "ips_relative_error":(r.ips_value-ref)/ref,"snips":r.snips_value,
                 "snips_relative_error":(r.snips_value-ref)/ref,"max_weight":r.max_weight,
                 "p99_weight":r.p99_weight,"p999_weight":r.p999_weight,
                 "ess":r.effective_sample_size,"ess_fraction":r.effective_sample_fraction,
                 "top_1pct_weight_share":r.top_1pct_weight_share,
                 "top_01pct_weight_share":r.top_01pct_weight_share,
                 "top_001pct_weight_share":r.top_001pct_weight_share,
                 "top_1pct_weighted_reward_share":r.top_1pct_weighted_reward_share}
            for c,v in r.clipped_ips_values.items():
                row[f"clipped_ips_{c:g}"]=v; row[f"clipped_ips_{c:g}_relative_error"]=(v-ref)/ref
            w.writerow(row)
    metrics={r.campaign:{"ips":r.ips_value,"reference_ctr":r.on_policy_reference_ctr,
             "snips":r.snips_value,"max_weight":r.max_weight,"ess_fraction":r.effective_sample_fraction,
             "top_1pct_weight_share":r.top_1pct_weight_share,
             "top_1pct_weighted_reward_share":r.top_1pct_weighted_reward_share,
             "clipped_ips":r.clipped_ips_values} for r in results}
    (a.artifact_dir/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n")
    lines=["# v1.2 OBD Real-Data Results","",
           "BTS logs evaluate Uniform Random; Random CTR is a separate on-policy empirical reference, not exact truth.","",
           "| Campaign | IPS | Random CTR | Rel. error | SNIPS | ESS frac. | Max weight |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        ref=r.on_policy_reference_ctr
        lines.append(f"| {r.campaign} | {r.ips_value:.8f} | {ref:.8f} | {(r.ips_value-ref)/ref:.2%} | {r.snips_value:.8f} | {r.effective_sample_fraction:.2%} | {r.max_weight:.1f} |")
    lines += ["","Clipped IPS is a bias-variance sensitivity analysis. ESS is a weight-concentration heuristic, not literal inferential sample size.",""]
    (a.artifact_dir/"RESULTS.md").write_text("\n".join(lines))
    print(f"wrote {a.artifact_dir/'summary.csv'}")
    print(f"wrote {a.artifact_dir/'metrics.json'}")
    print(f"wrote {a.artifact_dir/'RESULTS.md'}")
if __name__=="__main__": main()
