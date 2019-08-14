import os
import random
import subprocess
import numpy as np

seeds = [0, 1, 2, 3, 4]

def parse_all_seeds(experiment_name, seeds):
    acc = []
    norm_ED = []
    int_dist = []
    for seed in seeds:
        with open(f"./result/{experiment_name}-seed-{seed}/log_evaluation.txt", "r") as f:
            acc.append(float(f.readline()))
            norm_ED.append(float(f.readline()))
            int_dist.append(float(f.readline()))
    mean_acc = np.mean(np.array(acc))
    mean_norm_ED = np.mean(np.array(norm_ED))
    mean_int_dist = np.mean(np.array(int_dist))
    acc_var = np.std(np.array(acc))
    norm_ED_var = np.std(np.array(norm_ED))
    int_dist_var = np.std(np.array(int_dist))

    return mean_acc, acc_var, mean_norm_ED, norm_ED_var, mean_int_dist, int_dist_var

exp_name = 'VS'
mean_acc, acc_var, mean_norm_ED, norm_ED_var, mean_int_dist, int_dist_var = parse_all_seeds(exp_name, seeds)
print(f'{exp_name} :')
print(f'Acc: {mean_acc} +- {acc_var}\nNorm ED: {mean_norm_ED} +- {norm_ED_var}\nInt distance: {mean_int_dist} +- {int_dist_var}\n')

exp_name = 'VS-matching'
mean_acc, acc_var, mean_norm_ED, norm_ED_var, mean_int_dist, int_dist_var = parse_all_seeds(exp_name, seeds)
print(f'{exp_name} :')
print(f'Acc: {mean_acc} +- {acc_var}\nNorm ED: {mean_norm_ED} +- {norm_ED_var}\nInt distance: {mean_int_dist} +- {int_dist_var}\n')

exp_name = 'VS-ft'
mean_acc, acc_var, mean_norm_ED, norm_ED_var, mean_int_dist, int_dist_var = parse_all_seeds(exp_name, seeds)
print(f'{exp_name} :')
print(f'Acc: {mean_acc} +- {acc_var}\nNorm ED: {mean_norm_ED} +- {norm_ED_var}\nInt distance: {mean_int_dist} +- {int_dist_var}\n')

exp_name = 'VS-ft-matching'
mean_acc, acc_var, mean_norm_ED, norm_ED_var, mean_int_dist, int_dist_var = parse_all_seeds(exp_name, seeds)
print(f'{exp_name} :')
print(f'Acc: {mean_acc} +- {acc_var}\nNorm ED: {mean_norm_ED} +- {norm_ED_var}\nInt distance: {mean_int_dist} +- {int_dist_var}\n')

exp_name = 'VS-CBN'
mean_acc, acc_var, mean_norm_ED, norm_ED_var, mean_int_dist, int_dist_var = parse_all_seeds(exp_name, seeds)
print(f'{exp_name} :')
print(f'Acc: {mean_acc} +- {acc_var}\nNorm ED: {mean_norm_ED} +- {norm_ED_var}\nInt distance: {mean_int_dist} +- {int_dist_var}')

exp_name = 'VS-CBN-matching'
mean_acc, acc_var, mean_norm_ED, norm_ED_var, mean_int_dist, int_dist_var = parse_all_seeds(exp_name, seeds)
print(f'{exp_name} :')
print(f'Acc: {mean_acc} +- {acc_var}\nNorm ED: {mean_norm_ED} +- {norm_ED_var}\nInt distance: {mean_int_dist} +- {int_dist_var}')