import numpy as np
import pandas as pd
from Simulation_final import *
import scipy as sp
import pickle as pkl

from time import time
from joblib import Parallel, delayed
from itertools import product
from tqdm import tqdm

mu1 = np.array([0,1])
mu2 = np.array([1,0])
N1 = 50
N2 = 50
sigma1 = 0.05
sigma2 = 0.05
a = 2
b = 5
acc_prob=0.5
beta=10

def run(rec_how, acc_how, intervention_end=None):
    if intervention_end == None:
        intervention_time = []
        intervention_end = 'na'
    else:
        intervention_time = list(range(50, intervention_end))
        
    if acc_how == 'constant':
        acc_prob = 0.5
    else:
        acc_prob = 30
    
    conclusion = Conclusion(list(range(5)))
    grp_info = [{'mean': mu1, 'variance': sigma1*np.identity(2), 'size': N1},
                {'mean': mu2, 'variance': sigma2*np.identity(2), 'size': N2}]
    conclusion.run_experiments(grp_info, plot=False, init_how='embedding', Ns=N1+N2, Nf=10, 
                            node_step=5, total_step=400, acc_how=acc_how,acc_prob=acc_prob, a=a, b=b,
                            beta=beta,
                            p2_prob=0.5,
                            ng_how='embedding', 
                            intervention_time=intervention_time,
                            rec_how=rec_how,
                            node_removal=False,
                            edge_removal=True,
                            freq=5, record_each_run=False, rec_sample_fraction=0.1)
    conclusion.experiments = None
    fname='exp_delayed_effects_edge_removal/{}_{}_{}.pkl'.format(rec_how, acc_how, intervention_end)
    with open(fname, 'wb') as f:
        pkl.dump([rec_how, acc_how, intervention_end, conclusion], f)
        
rec_how  = ['embedding', 'random_fof']
acc_how = ['constant', 'embedding']
intervention_end = list(range(50, 401, 25))

settings = list(product(rec_how, acc_how, intervention_end))
Parallel(n_jobs=32)(delayed(run)(*setting) for setting in tqdm(settings))
    
    
    
    
    
def run(rec_how, acc_how, intervention_end=None):
    if intervention_end == None:
        intervention_time = []
        intervention_end = 'na'
    else:
        intervention_time = list(range(50, intervention_end))
        
    if acc_how == 'constant':
        acc_prob = 0.5
    else:
        acc_prob = 30
    
    conclusion = Conclusion(list(range(5)))
    grp_info = [{'mean': mu1, 'variance': sigma1*np.identity(2), 'size': N1},
                {'mean': mu2, 'variance': sigma2*np.identity(2), 'size': N2}]
    conclusion.run_experiments(grp_info, plot=False, init_how='embedding', Ns=N1+N2, Nf=10, 
                            node_step=5, total_step=400, acc_how=acc_how,acc_prob=acc_prob, a=a, b=b,
                            beta=beta,
                            p2_prob=0.5,
                            ng_how='embedding', 
                            intervention_time=intervention_time,
                            rec_how=rec_how,
                            node_removal=True,
                            freq=5, record_each_run=False, rec_sample_fraction=0.1)
    conclusion.experiments = None
    fname='exp_delayed_effects_node_removal/{}_{}_{}.pkl'.format(rec_how, acc_how, intervention_end)
    with open(fname, 'wb') as f:
        pkl.dump([rec_how, acc_how, intervention_end, conclusion], f)
        
rec_how  = ['embedding', 'random_fof']
acc_how = ['constant', 'embedding']
intervention_end = list(range(50, 401, 25))

settings = list(product(rec_how, acc_how, intervention_end))
Parallel(n_jobs=32)(delayed(run)(*setting) for setting in tqdm(settings))
    