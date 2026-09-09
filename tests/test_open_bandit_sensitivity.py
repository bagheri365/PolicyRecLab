import csv
from pathlib import Path
from policyreclab.experiments.open_bandit_sensitivity import run_bts_to_random_sensitivity

def _write(p:Path,rows):
    with p.open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=["item_id","position","click","propensity_score"]); w.writeheader()
        for a,pos,r,ps in rows:w.writerow({"item_id":a,"position":pos,"click":r,"propensity_score":ps})

def test_tail_and_snips(tmp_path):
    b=tmp_path/"b.csv"; r=tmp_path/"r.csv"
    _write(b,[(0,1,1,.5),(1,1,0,.25),(2,1,1,.01),(3,1,0,.25)])
    _write(r,[(0,1,1,.25),(1,1,0,.25),(2,1,1,.25),(3,1,1,.25)])
    x=run_bts_to_random_sensitivity(bts_csv=b,random_csv=r,campaign="all",n_actions=4,clip_thresholds=(2.,10.))
    assert x.max_weight==25.; assert x.snips_value!=x.ips_value
    assert x.clipped_ips_values[2.]<x.ips_value
    assert 0<x.effective_sample_fraction<1

def test_extreme_tail_share(tmp_path):
    b=tmp_path/"b.csv"; r=tmp_path/"r.csv"
    _write(b,[(0,1,0,.5)]*999+[(1,1,1,.0001)])
    _write(r,[(0,1,1,.5),(1,1,0,.5)])
    x=run_bts_to_random_sensitivity(bts_csv=b,random_csv=r,campaign="all",n_actions=2)
    assert x.top_001pct_weight_share>.8
    assert x.top_1pct_weighted_reward_share==1.
