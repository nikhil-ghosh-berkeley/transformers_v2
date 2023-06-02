import os
from joblib import Parallel, delayed

def download_experiment(main_exp_name, seed):
    cmd = f'amlt results download -I "events.out.tfevents.*" {main_exp_name}_seed_{seed} --output "results/"'
    os.system(cmd)

main_exp_name = "qqp_gpt2"
# seed_list = list(range(1, 11))
seed_list = [9]
n_jobs = min(24, len(seed_list))
results = Parallel(n_jobs=n_jobs)(delayed(download_experiment)(main_exp_name, seed) for seed in seed_list)    
    

