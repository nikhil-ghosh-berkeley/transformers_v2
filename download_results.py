import os
from joblib import Parallel, delayed

def download_experiment(exp):
    cmd = f'amlt results download -I "events.out.tfevents.*" {exp} --output "results/"'
    os.system(cmd)

main_exp_name = "qqp"
seed_list = list(range(1, 9))
exps = [f"{main_exp_name}_seed_{seed}" for seed in seed_list]
n_jobs = min(1, len(exps))
results = Parallel(n_jobs=n_jobs)(delayed(download_experiment)(exp) for exp in exps)    
    
# exps = ["qqp_subsamp_low_lr_high", "qqp_subsamp_low_lr_high2",  "qqp_subsamp_high_lr_low", "qqp_subsamp_high_lr_low2", "qqp_subsamp_high_lr_low3",  "qqp_subsamp_low_lr_low", "qqp_subsamp_low_lr_low2"]
# n_jobs = min(24, len(exps))
# results = Parallel(n_jobs=n_jobs)(delayed(download_experiment)(exp) for exp in exps)   
