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
from src.dataloader import read_data_mimo

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

H_num = 10
X_num = 10
num_rand = H_num * X_num

def get_parameter(SNR, N, K, config):
    config["num_ls"] = 4
    num_epochs = 3
    max_range = 150
    if N == 800 and K == 800:
        if SNR == 2:
            config["num_ls"] = 3
            num_epochs = 1
        elif SNR == 4:
            config["num_ls"] = 3
            num_epochs = 2
        elif SNR == 6:
            config["num_ls"] = 5
            num_epochs = 5
        elif SNR == 8:
            config["num_ls"] = 7
            num_epochs = 3
        elif SNR == 10:
            config["num_ls"] = 5
            num_epochs = 2
        elif SNR == 12:
            config["num_ls"] = 4
            num_epochs = 2
    if N == 1000 and K == 1000:
        if SNR == 2:
            config["num_ls"] = 2
            num_epochs = 2
        elif SNR == 4:
            config["num_ls"] = 3
            num_epochs = 2
        elif SNR == 6:
            config["num_ls"] = 7
            num_epochs = 4
        elif SNR == 8:
            config["num_ls"] = 8
            num_epochs = 4
        elif SNR == 10:
            config["num_ls"] = 7
            num_epochs = 2
        elif SNR == 12:
            config["num_ls"] = 3
            num_epochs = 3
    if N == 1200 and K == 1200:
        if SNR == 2:
            config["num_ls"] = 3
            num_epochs = 1
        elif SNR == 4:
            config["num_ls"] = 3
            num_epochs = 2
        elif SNR == 6:
            config["num_ls"] = 7
            num_epochs = 3
        elif SNR == 8:
            config["num_ls"] = 6
            num_epochs = 4
            max_range = 100
        elif SNR == 10:
            config["num_ls"] = 4
            num_epochs = 4
        elif SNR == 12:
            config["num_ls"] = 5
            num_epochs = 2
    config["max_epoch_num"] = num_epochs * config["sample_epoch_num"]
    return config, max_range

def simulation(mimo_size, snr):
    with open("../config/mimo_default.yaml") as f:
        config = yaml.safe_load(f)
    num_nodes = 2*mimo_size
    max_range = 150
    config["num_ls"] = 4  # local search epoch
    config, max_range = get_parameter(snr, mimo_size, mimo_size, config)  # get paramter

    rand_seeds = list(range(0, num_rand))

    record = []
    total_time = 0

    for r_seed in rand_seeds:

        data = read_data_mimo(mimo_size, mimo_size, snr, X_num, r_seed)

        total_start_time = time.perf_counter()
        _, _, now_best, now_best_info = mcpg_solver(num_nodes, config, data)
        now_best_info = now_best_info * 2 - 1
        total_end_time = time.perf_counter()
        total_time += total_end_time - total_start_time
        # get average information
        best_sort = torch.argsort(now_best, descending=True)
        total_best_info = torch.squeeze(now_best_info[:, best_sort[0]])
        for i0 in range(max_range):
            total_best_info += torch.squeeze(now_best_info[:, best_sort[i0]])
        total_best_info = torch.sign(total_best_info)
        record.append((total_best_info != data[2]).sum().item()/num_nodes)

    return min(record),  sum(record)/num_rand, total_time/num_rand

print("NAME\t\tMEAN BER\tTIME")
for qsize in [180, 200, 800, 1200]:
    for snr in [2, 4, 6, 8, 10, 12]:
        best_res, mean_res, ave_time = simulation(qsize, snr)
        print("{}-{}\t\t{:.6f}\t{:.3f}".format(qsize, snr, mean_res, ave_time))
        
