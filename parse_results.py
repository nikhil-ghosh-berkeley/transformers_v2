import amlt
from tbparse import SummaryReader
import os
import pandas as pd
import json
import logging
from tqdm import tqdm
from joblib import Parallel, delayed

def add_back_underscore(name):
    if name == "sampleprob":
        return "sample_prob"
    if name == "subsampratio":
        return "subsamp_ratio"
    return name


def job_name_to_hp(exp_name, job_name):
    prefix = f"sweep_{exp_name}_"
    job_name = job_name[len(prefix) :]
    hp = job_name.split("_")
    hp_dict = {}
    for i in range(0, len(hp), 2):
        hp_dict[add_back_underscore(hp[i])] = float(hp[i + 1])
    return hp_dict

def custom_logger(logger_name, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(asctime)s %(levelname)s %(message)s"
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    filename = f"{logger_name}.log"
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def parse_experiment(exp_name):
    print(f"starting {exp_name}!")
    exp = amlt.active_project().experiments.get(name=exp_name)
    logger = custom_logger(exp_name)

    metric_names = [
        ("eval/accuracy", "eval_acc"),
        ("train/loss", "neg_train_loss"),
        ("train/train_accuracy", "train_acc"),
    ]
    df_exp = {k[1]: [] for k in metric_names}

    for job_i in tqdm(range(len(exp.jobs))):
        job = exp.jobs[job_i]

        if 0 < job_i:
            run = job.run
            tags = run.get_tags()
            job_name = job.config.name
            log_dir = os.path.join(exp_name, job_name)

            if "hyperparameters" in tags:
                hyperparameters = json.loads(tags["hyperparameters"])
            else:
                hyperparameters = job_name_to_hp(exp_name, job_name)
            # breakpoint()
            if os.path.exists(log_dir):
                reader = SummaryReader(log_dir, pivot=True)
                logger.info(
                    f"processing job {job_i}, num rows={len(reader.scalars)}"
                )
                for metric_name in metric_names:
                    name = metric_name[0]
                    if name not in reader.scalars.columns:
                        logger.warning(f"metric {name} not found in job {job_i}: {job_name}")
                        continue
                    df_job = reader.scalars[[name]]
                    df_job[name] = df_job[name].apply(
                        lambda x: x[0] if isinstance(x, list) else x
                    )
                    for k, v in hyperparameters.items():
                        df_job[k] = v
                    if "loss" in name:
                        df_job[name] = -df_job[name]
                    max_idx = df_job[name].idxmax()
                    df_exp[metric_name[1]].append(df_job.iloc[[max_idx]])
            else:
                logger.warning(f"log_dir {log_dir} not found for job {job_i}")

    for k, v in df_exp.items():
        df = pd.concat(v)
        df.to_pickle(f"{exp_name}_max_{k}.pkl")


def parse_seeded_experiment(main_exp_name, seed):
    exp_name = f"{main_exp_name}_seed_{seed}"
    parse_experiment(exp_name)
    

def main():
    main_exp_name = "qqp_gpt2"
    # seed_list = list(range(1, 11))
    seed_list = [9]
    # Set the number of jobs to use
    n_jobs = min(24, len(seed_list))
    results = Parallel(n_jobs=n_jobs)(delayed(parse_seeded_experiment)(main_exp_name, seed) for seed in seed_list)

if __name__ == "__main__":
    main()