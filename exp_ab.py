import numpy as np
from simulation import *
import pickle as pkl

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
rerun=True

def run(rec_how, acc_how, intervention_end=None, node_removal=False, edge_removal=False, treatment_probability=0.5,rerun=False):
    
    folder_name = 'experiments/exp_ab_beta10'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if isinstance(treatment_probability, list):
        treatment = f'group{treatment_probability[0]}'
    else:
        treatment = treatment_probability
    fname = f'{folder_name}/{rec_how}_{acc_how}_{intervention_end}_{node_removal}_{edge_removal}_{treatment}.pkl'
    
    if os.path.exists(fname) and not rerun:
        print(f'{fname} exists, skip') 
        return

    print(f'running {fname}')
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
                            is_ab_test=True, treatment_probability=treatment_probability,
                            freq=5, record_each_run=False, rec_sample_fraction=0.1)
    conclusion.experiments = None

    with open(fname, 'wb') as f:
        pkl.dump([rec_how, acc_how, intervention_end, conclusion], f)
        print(f'{fname} saved')
        
rec_how  = ['random_fof']
acc_how = ['constant', 'embedding']
intervention_end = [200, 400]
node_removal = [False]
edge_removal = [False]
treatment_probability = [[1], 0.5]

settings = list(product(rec_how, acc_how, intervention_end, node_removal, edge_removal, treatment_probability, [rerun]))
Parallel(n_jobs=8)(delayed(run)(*setting) for setting in tqdm(settings))
