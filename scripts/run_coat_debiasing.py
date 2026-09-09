from __future__ import annotations
import argparse
from pathlib import Path
from policyreclab.datasets.mnar_ratings import load_coat_matrix
from policyreclab.experiments.mnar_debiasing import run_mnar_mean_debiasing

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,default=Path("data/coat"))
    a=p.parse_args()
    train=load_coat_matrix(a.data_root/"train.ascii")
    test=load_coat_matrix(a.data_root/"test.ascii")
    result=run_mnar_mean_debiasing(dataset="coat",observational=train,randomized=test,n_users=290,n_items=300)
    for k,v in result.__dict__.items(): print(f"{k}: {v}")
if __name__=="__main__": main()
