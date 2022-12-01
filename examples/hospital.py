from pyinla.model import *
import numpy as np
import pandas as pd

formula = 'r ~ f(hospital, model="iid", hyper=prior_prec)'
prior_prec = dict(prec=dict(prior="pc.prec", param=np.array([1.0, 0.01])))
priors = dict(prior_prec=prior_prec)
d = pd.read_csv("./data/hospital.csv", delim_whitespace=True)
data = pd_to_dict(d) | priors
result = inla(
    formula,
    data,
    family="binomial",
    n_trials=data["n"],
    control_compute=dict(dic=True),
    control_predictor=dict(compute=True),
)
print(summary(result))
