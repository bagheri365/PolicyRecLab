from pathlib import Path
import numpy as np
from policyreclab.datasets.mnar_ratings import RatingTriples,load_coat_matrix,load_rating_triples
from policyreclab.experiments.mnar_debiasing import run_mnar_mean_debiasing

def test_coat_matrix_loader(tmp_path:Path):
    p=tmp_path/"x.ascii"; np.savetxt(p,np.array([[5,0],[0,2]]),fmt="%d")
    x=load_coat_matrix(p)
    assert x.n_ratings==2
    assert sorted(x.ratings.tolist())==[2.,5.]

def test_triple_loader(tmp_path:Path):
    p=tmp_path/"r.txt"; p.write_text("1 2 5\n2 3 1\n")
    x=load_rating_triples(p)
    assert x.n_ratings==2

def test_mnar_result_keeps_randomized_reference_separate():
    obs=RatingTriples(np.array([0,0,1]),np.array([0,0,1]),np.array([5.,4.,1.]))
    rnd=RatingTriples(np.array([0,1]),np.array([0,1]),np.array([2.,2.]))
    r=run_mnar_mean_debiasing(dataset="toy",observational=obs,randomized=rnd,n_users=2,n_items=2)
    assert r.randomized_reference_mean==2.
    assert r.naive_observational_mean>r.randomized_reference_mean
    assert r.propensity_source=="estimated_item_frequency"
    assert "not production contextual-bandit OPE" in r.interpretation

def test_supplied_item_propensities_are_validated():
    obs=RatingTriples(np.array([0]),np.array([0]),np.array([5.]))
    rnd=RatingTriples(np.array([0]),np.array([0]),np.array([3.]))
    try:
        run_mnar_mean_debiasing(dataset="toy",observational=obs,randomized=rnd,n_users=1,n_items=1,item_propensities=np.array([0.]))
    except ValueError as e:
        assert "propensities" in str(e)
    else: raise AssertionError("expected invalid propensity failure")
