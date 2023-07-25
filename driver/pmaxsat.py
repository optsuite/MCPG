import sys
sys.path.append("..")
sys.path.append("../src")
import torch
import numpy as np
import yaml
import time
from src.mcpg_solver import mcpg_solver
from src.dataloader import dataloader_select


torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def driver(prob_dict, repeat=10):
    print("NAME\t\t\t\tMAX OBJ\tMEAN OBJ\tMEAN GAP\tAVE TIME")
    for key in prob_dict:
        prob_name = prob_dict[key]["problem name"]
        path = prob_dict[key]["path"]
        config_file = prob_dict[key]["config"]
        with open(config_file) as f:
            config = yaml.safe_load(f)
        dataloader = dataloader_select(config["problem_type"])
        data, nvar = dataloader(path)

        num_epochs = 35
        config["num_ls"] = 2
        if nvar>=700:
            config["total_mcmc_num"] = 900
        elif nvar>=1000:
            config["total_mcmc_num"] = 1200
        else: 
            config["total_mcmc_num"] = 600 
            num_epochs = 50 
        config["repeat_times"] = 80 

        config["max_epoch_num"] = (num_epochs-1) * config["sample_epoch_num"]+1

        res_list = []
        time_list = []
        for i in range(repeat):
            torch.manual_seed(30 + i)
            start_time = time.perf_counter()
            res, _, _, _ = mcpg_solver(nvar, config, data, verbose=False)
            end_time = time.perf_counter()
            if res > data.pdata[5] * data.pdata[6]:
                res -= data.pdata[5] * data.pdata[6]
            else: 
                res = res//data.pdata[5]-data.pdata[6]
            res_list.append(res)
            time_list.append(end_time - start_time)


        print("{}\t{}\t{:.1f}\t\t{:.2f}\t\t{:.1f}".format(prob_name,
                                                  np.max(res_list),
                                                  np.mean(res_list),
                                                  (np.max(res_list) - np.mean(res_list)) / np.max(res_list) * 100,
                                                  np.mean(time_list)))    

prob_ind = list(range(1, 30))
prob_name = ["clq1-cv260c1040l2g{}".format(ind) for ind in prob_ind]
prob_info = {}
for name in prob_name:
    info = {}
    info["problem name"] = name
    info["path"] = "../data/partial_sat/{}.wcnf".format(name)
    info["config"] = "../config/maxsat_default.yaml"
    prob_info[name] = info

driver(prob_info, repeat=5)