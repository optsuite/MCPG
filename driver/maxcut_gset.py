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

upper_bound = {
    "G22":13359,
    "G23":13344,
    "G24":13337,
    "G25":13340,
    "G26":13328,
    "G27":3341,
    "G28":3298,
    "G29":3405,
    "G30":3413,
    "G31":3310,
    "G32":1410,
    "G33":1382,
    "G34":1384,
    "G35":7687,
    "G36":7680,
    "G37":7691,
    "G38":7688,
    "G39":2408,
    "G40":2400,
    "G41":2405,
    "G42":2481,
    "G43":6660,
    "G44":6650,
    "G45":6654,
    "G46":6649,
    "G47":6657,
    "G55":10296,
    "G56":4012,
    "G57":3492,
    "G58":19263,
    "G59":6078,
    "G60":14176,
    "G61":5789,
    "G62":4868,
    "G63":26997,
    "G64":8735,
    "G65":5558,
    "G66":6360,
    "G67":6940,
    "G70":9595,
    "G72":6998,
    "G77":9926,
    "G81":14030
}

def driver(prob_dict, repeat=10):
    print("NAME\tUB\t\tBEST OBJ\tMIN GAP\tMEAN OBJ\tMEAN GAP\tAVE TIME")
    for key in prob_dict:
        prob_name = prob_dict[key]["problem name"]
        ub = prob_dict[key]["UB"]
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
        print("{}\t\t{}\t{:.1f}\t\t{:.2f}\t{:.1f}\t\t{:.2f}\t\t{:.1f}".format(prob_name,
                                                    ub,
                                                    best_res,
                                                    (ub - best_res) / best_res * 100,
                                                    mean_res,
                                                    (ub - mean_res) / mean_res * 100,
                                                    np.mean(time_list)))

prob_ind = list(range(22,48))
prob_name = ["G{}".format(ind) for ind in prob_ind]
prob_info = {}
for name in prob_name:
    info = {}
    info["problem name"] = name
    info["path"] = "../data/graph/{}.txt".format(name)
    info["config"] = "../config/maxcut_default.yaml"
    info["UB"] = upper_bound[name]
    prob_info[name] = info

driver(prob_info, repeat=20)
