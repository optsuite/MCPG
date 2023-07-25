"""
Copyright (c) 2024 Cheng Chen, Ruitao Chen, Tianyou Li, Zaiwen Wen
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that
the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
   following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
   and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
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
    print("NAME\t\t\t\t\tBEST OBJ\tP-RATIO\tMEAN OBJ\tP-RATIO\tAVE TIME")
    for key in prob_dict:
        prob_name = prob_dict[key]["problem name"]
        path = prob_dict[key]["path"]
        config_file = prob_dict[key]["config"]
        nvar_info = prob_dict[key]["nvar"]
        degree_info = prob_dict[key]["d"]
        with open(config_file) as f:
            config = yaml.safe_load(f)
        dataloader = dataloader_select(config["problem_type"])
        data, nvar = dataloader(path)
        res_list = []
        time_list = []
        pratio_list = []
        for i in range(repeat):
            torch.manual_seed(30 + i)
            start_time = time.perf_counter()
            res, _, _, _ = mcpg_solver(nvar, config, data, verbose=False)
            end_time = time.perf_counter()
            res_list.append(res)
            pratio_list.append((res / nvar_info - degree_info / 4) / np.sqrt(degree_info / 4))
            time_list.append(end_time - start_time)
        best_res = np.max(res_list)
        best_pratio = np.max(pratio_list)
        mean_res = np.mean(res_list)
        mean_pratio = np.mean(pratio_list)
        print("{}\t{}\t\t{:.3f}\t{:.2f}\t{:.3f}\t{:.2f}".format(prob_name,
                                                    best_res,
                                                    best_pratio,
                                                    mean_res,
                                                    mean_pratio,
                                                    np.mean(time_list)))
        
prob_info = {}
for nvar in [10000, 20000, 30000, 40000, 50000]:
    name = "regular_n_{}_d_5_0".format(nvar)
    info = {}
    info["problem name"] = name
    info["path"] = "../data/regular_graph/{}.txt".format(name)
    info["config"] = "../config/maxcut_large.yaml"
    info["d"] = 5
    info["nvar"] = nvar
    prob_info[name] = info

driver(prob_info, repeat=5)
