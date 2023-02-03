import numpy as np
from simulation import *
import csv

from joblib import Parallel, delayed
from itertools import product
from tqdm import tqdm

file_name = 'experiments/exp_robustness.json'
num_group = 2
record_time = range(0, 400, 50)

# mu, N, sigma are list of lists, each inner list contains settings for different groups
mu = [[[0, 1], [1, 0]]]
N = [[50, 50]]
sigma = [[0.05, 0.05]]
a = [2]
b = [5]
acc_prob = [0.5]
beta = [10]

rec_how  = ['embedding', 'random_fof']
acc_how = ['constant', 'embedding']
intervention_end = [50, 100, 200, 300, 400]
node_removal = [False]
edge_removal = [False]

is_ab_test = [True]
treatment_probability = [0.1]
treatment_size = [None]
treatment_time = [50]

metric_names = ['added_nodes', 'nodes', 'edges', 'avg_degree', 'degree_variance', 'bi_frac', 'global_clustering', 'avg_age', 'avg_nn', 'gini_coeff', 'num_rec', 'num_acc']
for name in ['phase0', 'phase1', 'phase2_unmediated', 'phase2_mediated', 'phase3']:
    metric_names.append(name)
    metric_names.append(f"{name}_mono")
    metric_names.append(f"{name}_bi")
for i in range(num_group):
    metric_names.append(f"nodes_{i}")
    metric_names.append(f"homophily_{i}")
    metric_names.append(f"avg_degree_{i}")
    metric_names.append(f"degree_variance_{i}")
    metric_names.append(f"global_clustering_{i}")
    metric_names.append(f"gini_coeff_{i}")

# for csv saving
# setting_names = ['mu', 'N', 'sigma', 'a', 'b', 'acc_prob', 'beta', 'rec_how', 'acc_how', 'intervention_end', 'node_removal', 'edge_removal', 'is_ab_test', 'treatment_probability', 'treatment_size', 'treatment_time']

# metrics = []
# for metric_name in metric_names:
#     for t in record_time:
#         metrics.extend([f"{metric_name}_{t}_avg", f"{metric_name}_{t}_ci"])

# if not os.path.exists(file_name):
#     header = setting_names + metrics

#     if not os.path.exists('experiments'):
#         os.makedirs('experiments')
#     with open(file_name, 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)

def run(mu, N, sigma, a, b, acc_prob, beta, rec_how, acc_how, intervention_end=None, node_removal=False, edge_removal=False, is_ab_test=False, treatment_probability=1, treatment_size=None, treatment_time=50):
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
    grp_info = [{'mean': [np.array(m) for m in mu[i]], 'variance': sigma[i]*np.identity(2), 'size':N[i]} for i in range(len(N))]

    conclusion.run_experiments(grp_info, plot=False, init_how='embedding', Ns=sum(N), Nf=10, 
                            node_step=5, total_step=400, acc_how=acc_how,acc_prob=acc_prob, a=a, b=b,
                            beta=beta,
                            p2_prob=0.5,
                            ng_how='embedding', 
                            intervention_time=intervention_time,
                            rec_how=rec_how,
                            node_removal=node_removal,
                            edge_removal=edge_removal,
                            freq=5, record_each_run=False, rec_sample_fraction=0.1,
                            is_ab_test=is_ab_test, treatment_probability=treatment_probability, treatment_size=treatment_size, treatment_time=treatment_time)
    conclusion.experiments = None

    # for csv saving
    # settings = [mu, N, sigma, a, b, acc_prob, beta, rec_how, acc_how, intervention_end, node_removal, edge_removal, is_ab_test, treatment_probability, treatment_size, treatment_time]
    # metrics = []
    # for metric in metric_names:
    #     for t in record_time:
    #         metrics.append(conclusion.avg_values[metric][t//5])
    #         metrics.append(conclusion.ci_values[metric][t//5])
    # row = settings + metrics
    # with open(file_name, 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(row)

    # for json saving
    settings = [mu, N, sigma, a, b, acc_prob, beta, rec_how, acc_how, intervention_end, node_removal, edge_removal, is_ab_test, treatment_probability, treatment_size, treatment_time]

    metrics = {}
    for metric in metric_names:
        metrics[metric] = {'avg':[], 'ci':[]}
        for t in record_time:
            metrics[metric]['avg'].append(conclusion.avg_values[metric][t//5])
            metrics[metric]['ci'].append(conclusion.ci_values[metric][t//5])
    entry = {'settings': settings, 'metrics': metrics}
    
    if not os.path.isfile(file_name):
        with open(file_name, 'w') as f:
            json.dump([entry], f)
    else:
        with open(file_name) as feedsjson:
            feeds = json.load(feedsjson)

        feeds.append(entry)
        with open(file_name, mode='w') as f:
            json.dump(feeds, f)

settings = list(product(mu, N, sigma, a, b, acc_prob, beta, rec_how, acc_how, intervention_end, node_removal, edge_removal, is_ab_test, treatment_probability, treatment_size, treatment_time))

# filter out existing settings
if os.path.exists(file_name):
    with open(file_name) as feedsjson:
            existing_experiments = json.load(feedsjson)
            existing_settings = [exp['settings'] for exp in existing_experiments]
            print(existing_settings[0])
            settings = [s for s in settings if list(s) not in existing_settings]
Parallel(n_jobs=32)(delayed(run)(*setting) for setting in tqdm(settings))