from __future__ import annotations
import argparse
from pathlib import Path
from policyreclab.datasets.mnar_ratings import load_rating_triples
from policyreclab.experiments.mnar_debiasing import run_mnar_mean_debiasing

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True,help="Yahoo R3 user-selected ratings file")
    p.add_argument("--randomized",type=Path,required=True,help="Yahoo R3 randomly selected ratings file")
    a=p.parse_args()
    train=load_rating_triples(a.train)
    randomized=load_rating_triples(a.randomized)
    result=run_mnar_mean_debiasing(dataset="yahoo_r3",observational=train,randomized=randomized,n_users=15400,n_items=1000)
    for k,v in result.__dict__.items(): print(f"{k}: {v}")
if __name__=="__main__": main()
