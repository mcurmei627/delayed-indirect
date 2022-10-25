import numpy as np
import pandas as pd
from Simulation_final import *
import scipy as sp
import pickle as pkl

from time import time
from joblib import Parallel, delayed
from itertools import product
from tqdm import tqdm

a = 2
b = 5
beta=10

def run(setting, rec_how, acc_how, intervention_end=None, node_removal=False, edge_removal=False, p2_indirect=False):
    if setting == 'heterogeneity':
        mu1 = np.array([0,1.1])
        mu2 = np.array([1.1,0])
        N1 = 50
        N2 = 50
        sigma1 = 0.1
        sigma2 = 0.1
    
    elif setting == 'homogeneity':
        mu1 = np.array([0,1])
        mu2 = np.array([1,0])
        N1 = 50
        N2 = 50
        sigma1 = 0.01
        sigma2 = 0.01
    
    elif setting == 'minority_homophily':
        mu1 = np.array([0,1])
        mu2 = np.array([1.2,0])
        N1 = 60
        N2 = 40
        sigma1 = 0.05
        sigma2 = 0.01
    
    elif setting == 'majority_heterophily':
        mu1 = np.array([0,0.6])
        mu2 = np.array([0.3,1.3])
        N1 = 60
        N2 = 40
        sigma1 = 0.03
        sigma2 = 0.005
        
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
                            node_removal=node_removal,
                            edge_removal=edge_removal, 
                            p2_indirect=p2_indirect,
                            comp_grp_metrics=True,
                            freq=5, record_each_run=False, rec_sample_fraction=0.1)
    conclusion.experiments = None
    fname='group_effects/{}_{}_{}_{}_{}_{}_{}.pkl'.format(setting, rec_how, acc_how, intervention_end, node_removal, edge_removal, p2_indirect)
    with open(fname, 'wb') as f:
        pkl.dump([rec_how, acc_how, intervention_end, conclusion], f)

settings = ['heterogeneity', 'homogeneity', 'minority_homophily', 'majority_heterophily']        
rec_how  = ['embedding', 'random_fof']
acc_how = ['constant']
intervention_end = list(range(50, 401, 100))
node_removal = [True, False]
edge_removal = [True, False]
p2_indirect = [True, False]

settings = list(product(settings, rec_how, acc_how, intervention_end, node_removal, edge_removal, p2_indirect))
Parallel(n_jobs=64)(delayed(run)(*setting) for setting in tqdm(settings))
    