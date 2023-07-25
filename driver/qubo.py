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
    print("NAME\t\tBEST OBJ\tMEAN OBJ\tMEAN GAP\tAVE TIME")
    for key in prob_dict:
        prob_name = prob_dict[key]["problem name"]
        path = prob_dict[key]["path"]
        config_file = prob_dict[key]["config"]
        with open(config_file) as f:
            config = yaml.safe_load(f)
        dataloader = dataloader_select(config["problem_type"])
        data, nvar = dataloader(path)
        res_list = []
        time_list = []
        for i in range(repeat):
            torch.manual_seed(30 + i)
            start_time = time.perf_counter()
            res, _, _, _ = mcpg_solver(nvar, config, data, verbose=False)
            end_time = time.perf_counter()
            res_list.append(res)
            time_list.append(end_time - start_time)
        best_res = np.max(res_list)
        mean_res = np.mean(res_list)
        print("{}\t{}\t{:.1f}\t{:.2f}\t{:1f}".format(prob_name,
                              best_res,
                              mean_res, 
                              (best_res - mean_res) / mean_res * 100,
                              np.mean(time_list)))

prob_name = []
for nvar in [5000, 7000, 10000]:
    for ind in range(1, 6):
        prob_name.append("nbiq_{}_{}".format(nvar, ind))
prob_info = {}
for name in prob_name:
    info = {}
    info["problem name"] = name
    info["path"] = "../data/nbiq/{}.npy".format(name)
    info["config"] = "../config/qubo_default.yaml"
    prob_info[name] = info

driver(prob_info, repeat=5)