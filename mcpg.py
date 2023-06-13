import torch
import argparse
import yaml
import time
from sampling import mcpg_sampling_selector
from dataloader import dataloader_selector
from model import simple
from torch.distributions.bernoulli import Bernoulli
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("config_file", type=str,
                    help="input the configuration file for the mcpg solver")
parser.add_argument("problem_instance", type=str,
                    help="input the data file for the problem instance")

args = parser.parse_args()

with open(args.config_file) as f:
    config = yaml.safe_load(f)
mcpg_sampling = mcpg_sampling_selector(config['problem type'])
dataloader = dataloader_selector(config['problem_type'])

with open(args.problem_instance) as f:
    data = dataloader(f)

total_mcmc_num = config['total_mcmc_num']
repeat_times = config['repeat_times']
num_epochs = config['num_epochs']
num_ls = config['num_ls']
num_node = data['num_node']
lr = config['lr']
regular = config['regular']
sample_epoch_num = config['sample_epoch_num']

num_nodes = data.num_nodes
mcmc_transfer_num_base = int(num_nodes/10)
max_epoch_num = num_epochs * sample_epoch_num
reset_epoch_num = 10*sample_epoch_num


total_max_cut = 0
now_max_cut = 0
now_max_info = []
total_max_info = []

loss = "loss"


net = simple(32, 8, num_nodes, deltas=1)
net.to(device).reset_parameters()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

samples = None
start_samples = None
stop_step = torch.zeros(total_mcmc_num)

start_time = time.time()
for epoch in range(max_epoch_num):

    if epoch % reset_epoch_num == 0:
        net.to(device).reset_parameters()
        net.train()
        if epoch <= 0:
            retdict = net(data, regular, None, None)
        else:
            retdict = net(data, regular, start_samples, value)

        retdict[loss][0].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        # get start samples
    if epoch == 0:

        probs = (torch.zeros(num_nodes)+0.5).to(device)

        m = Bernoulli(probs)
        samples = m.sample([total_mcmc_num*repeat_times])
        samples = samples.detach().to(device)
        tensor_probs = samples.t()
        temp_max, temp_max_info, ls_samples_temp, temp_start_samples, value = mcpg_sampling(
            data, tensor_probs, probs, 0)
        now_max_cut = temp_max
        now_max_info = temp_max_info
        total_max_cut = temp_max
        total_max_info = temp_max_info
        tensor_probs = temp_max_info.clone()
        tensor_probs = tensor_probs.repeat(1, repeat_times)
        samples = ls_samples_temp.t().to(device)
        start_samples = temp_start_samples.t().to(device)

    if epoch % sample_epoch_num == 0 and epoch > 0:
        probs = retdict["output"][0]
        probs = probs.detach()
        temp_max, temp_max_info, ls_samples_temp, start_samples_temp, value = mcpg_sampling(
            data, tensor_probs, probs, mcmc_transfer_num_base)

        # update now_max
        for i0 in range(total_mcmc_num):
            if temp_max[i0] > now_max_cut[i0]:
                now_max_cut[i0] = temp_max[i0]
                now_max_info[:, i0] = temp_max_info[:, i0]
                stop_step[i0] = 0
            else:
                stop_step[i0] += 1

        # update if min is too small
        now_max = max(now_max_cut).item()
        now_max_index = torch.argmax(now_max_cut)
        now_min = min(now_max_cut).item()
        now_min_index = torch.argmin(now_max_cut)
        now_max_cut[now_min_index] = now_max
        now_max_info[:, now_min_index] = now_max_info[:, now_max_index]
        temp_max_info[:, now_min_index] = now_max_info[:, now_max_index]

        # get max graph samples
        tensor_probs = temp_max_info.clone()
        tensor_probs = tensor_probs.repeat(1, repeat_times)
        # get samples
        samples = ls_samples_temp.t().to(device)
        start_samples = start_samples_temp.t()

    total_max = now_max_cut

print(total_max, "{:.2f}".format(time.time() - start_time))
