import torch

from model import simple
from dataloader import dataloader_select
from sampling import sampler_select, sample_initializer

def mcpg_solver(config, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = dataloader_select(config["problem_type"])
    sampler = sampler_select(config["problem_type"])

    data, nvar = dataloader(path)
    change_times = int(nvar/5)  # transition times for metropolis sampling


    net = simple(nvar)
    net.to(device).reset_parameters()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr_init'])

    start_samples = None
    for epoch in range(config['max_epoch_num']):

        if epoch % config['reset_epoch_num'] == 0:
            net.to(device).reset_parameters()
            regular = config['regular_init']

        net.train()
        if epoch <= 0:
            retdict = net(regular, None, None)
        else:
            retdict = net(regular, start_samples, value)

        optimizer.zero_grad()
        retdict["loss"][0].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        # get start samples
        if epoch == 0:
            probs = (torch.zeros(nvar)+0.5).to(device)
            tensor_probs = sample_initializer(
                config["problem_type"], probs, config, data = data)
            temp_max, temp_max_info, temp_start_samples, value = sampler(
                data, tensor_probs, probs, config['num_ls'], 0, config['total_mcmc_num'])
            now_max_res = temp_max
            now_max_info = temp_max_info
            tensor_probs = temp_max_info.clone()
            tensor_probs = tensor_probs.repeat(1, config['repeat_times'])
            start_samples = temp_start_samples.t().to(device)

        # get samples
        if epoch % config['sample_epoch_num'] == 0 and epoch > 0:
            probs = retdict["output"][0]
            probs = probs.detach()
            temp_max, temp_max_info, start_samples_temp, value = sampler(
                data, tensor_probs, probs, config['num_ls'], change_times, config['total_mcmc_num'])
            # print(temp_max)
            # update now_max
            for i0 in range(config['total_mcmc_num']):
                if temp_max[i0] > now_max_res[i0]:
                    now_max_res[i0] = temp_max[i0]
                    now_max_info[:, i0] = temp_max_info[:, i0]

            # update if min is too small
            now_max = max(now_max_res).item()
            now_max_index = torch.argmax(now_max_res)
            now_min = min(now_max_res).item()
            now_min_index = torch.argmin(now_max_res)
            now_max_res[now_min_index] = now_max
            now_max_info[:, now_min_index] = now_max_info[:, now_max_index]
            temp_max_info[:, now_min_index] = now_max_info[:, now_max_index]

            # select best samples
            tensor_probs = temp_max_info.clone()
            tensor_probs = tensor_probs.repeat(1, config['repeat_times'])
            # construct the start point for next iteration
            start_samples = start_samples_temp.t()
            print("o {:.3f}".format((max(now_max_res).item())))

        del(retdict)

    total_max = now_max_res

    print("output: ", max(total_max).item())
    return max(total_max).item(), now_max_info
