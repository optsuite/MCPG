import torch
from torch_scatter import scatter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mcpg_sampling_selector(problem_type):
    if problem_type == "maxcut":
        return mcpg_sampling_maxcut
    elif problem_type == "maxsat":
        return mcpg_sampling_maxsat
    elif problem_type == "MIMO":
        return mcpg_sampling_mimo
    elif problem_type == "cheeger cut":
        return mcpg_sampling_cheegercut
    else:
        raise(Exception("Unrecognized problem type {}".format(problem_type)))


def metro_sampling(probs, start_status, max_transfer_time, device):
    num_node = len(probs)
    num_chain = start_status.shape[1]
    index_col = torch.tensor(list(range(num_chain))).to(device)

    probs = probs.detach().to(device)
    samples = start_status.bool().to(device)
    count = 0
    t = 0
    for t in range(max_transfer_time*5):

        if count >= num_chain*max_transfer_time:
            break
        index_row = torch.randint(low=0, high=num_node, size=[
                                  num_chain], device=torch.device('cuda:0'))

        chosen_probs_base = probs[index_row]

        chosen_value = samples[index_row, index_col]

        chosen_probs = torch.where(
            chosen_value, chosen_probs_base, 1-chosen_probs_base)

        accept_rate = (1 - chosen_probs) / chosen_probs
        r = torch.rand(num_chain, device=torch.device('cuda:0'))
        is_accept = (r < accept_rate)
        samples[index_row, index_col] = torch.where(
            is_accept, ~chosen_value, chosen_value)

        count += is_accept.sum()

    return samples.float().to(device)


def mcpg_sampling_maxcut(data, start_result, probs, change_times, total_mcmc_num, num_ls):
    probs = probs.to(torch.device("cpu"))
    num_nodes = data.num_nodes
    edges = data.edge_index
    nlr_graph = edges[0]
    nlc_graph = edges[1]
    edge_weight = data.edge_attr
    edge_weight_sum = data.edge_weight_sum
    graph_probs = start_result.clone()
    # get probs
    graph_probs = metro_sampling(
        probs, graph_probs, change_times, device)
    start = graph_probs.clone()

    temp = graph_probs[data.sorted_degree_nodes[0]].clone()
    graph_probs += temp
    graph_probs = graph_probs % 2

    graph_probs = (graph_probs-0.5)*2+0.5

    # local search
    expected_cut = torch.zeros(graph_probs.size(dim=1))
    cnt = 0
    while True:
        cnt += 1
        for node_index in range(0, num_nodes):
            node = data.sorted_degree_nodes[node_index]
            neighbor_index = data.neighbors[node]
            neighbor_edge_weight = data.neighbor_edges[node]
            node_temp_v = torch.mm(
                neighbor_edge_weight, graph_probs[neighbor_index])
            node_temp_v = torch.squeeze(node_temp_v)
            node_temp_v += torch.rand(node_temp_v.shape[0],
                                      device=torch.device('cuda:0'))/4
            graph_probs[node] = (node_temp_v <
                                 data.weighted_degree[node]/2+0.125).int()
        if cnt >= num_ls:
            break

    expected_cut[:] = ((2 * graph_probs[nlr_graph.type(torch.long)][:] - 1)*(
        2 * graph_probs[nlc_graph.type(torch.long)][:] - 1) * edge_weight).sum(dim=0)

    expected_cut_reshape = torch.reshape(expected_cut, (-1, total_mcmc_num))
    index = torch.argmin(expected_cut_reshape, dim=0)
    for i0 in range(total_mcmc_num):
        index[i0] = i0 + index[i0]*total_mcmc_num
    max_cut = expected_cut[index]

    return ((edge_weight_sum-max_cut) / 2), graph_probs[:, index], graph_probs, start, (expected_cut-torch.mean(expected_cut)).to(device)


def mcpg_sampling_maxsat(
        pdata, ndata,
        start_result, probs,
        num_ls, change_times, total_mcmc_num):
    nvar, nclause, vi, ci, neg = pdata[0:5]
    raw_samples = start_result.clone()
    raw_samples = metro_sampling(
        probs, raw_samples, change_times, device)
    samples = raw_samples.t().clone()

    for cnt in range(num_ls):
        for index in range(nvar):
            i = ndata[3][index].item()
            cal_sample = samples[:, ndata[0][i]] * ndata[2][i]
            res_clause = scatter(cal_sample, ndata[1][i], reduce="max", dim=1)
            res_sample_old = torch.sum(res_clause, dim=1)
            samples[:, i] = - samples[:, i]
            cal_sample = samples[:, ndata[0][i]] * ndata[2][i]
            res_clause = scatter(cal_sample, ndata[1][i], reduce="max", dim=1)
            res_sample_new = torch.sum(res_clause, dim=1)
            #ind = (res_sample_new > res_sample_old)
            ind = (res_sample_new > res_sample_old +
                   torch.rand(res_sample_old.shape[0], device=device)-0.5)
            samples[:, i] = torch.where(ind, samples[:, i], -samples[:, i])

    cal_sample = samples[:, vi] * neg
    res_clause = scatter(cal_sample, ci, reduce="max", dim=1)
    res_sample = torch.sum(res_clause, dim=1)
    if len(pdata) == 7:
        res_sample += nclause-pdata[6]+pdata[5]*pdata[6]
    else:
        res_sample += nclause
    res_sample = res_sample/2

    res_sample_reshape = torch.reshape(res_sample, (-1, total_mcmc_num))
    index = torch.argmax(res_sample_reshape, dim=0)
    index = torch.tensor(list(range(total_mcmc_num)),
                         device=device) + index*total_mcmc_num
    max_res = res_sample[index]

    return max_res, samples.t()[:, index], samples.t(), raw_samples, -(res_sample - torch.mean(res_sample.float())).to(device)


def mcpg_sampling_cheegercut(data, start_result, probs, change_times, total_mcmc_num, num_ls):
    probs = probs.to(torch.device("cpu"))
    num_nodes = data.num_nodes
    edges = data.edge_index
    nlr_graph = edges[0]
    nlc_graph = edges[1]
    edge_weight = data.edge_attr
    edge_weight_sum = data.edge_weight_sum
    graph_probs = start_result.clone()
    # get probs
    graph_probs = metro_sampling(
        probs, graph_probs, change_times, device)
    start = graph_probs.clone()

    temp = graph_probs[data.sorted_degree_nodes[0]].clone()
    graph_probs += temp
    graph_probs = graph_probs % 2

    graph_probs = (graph_probs-0.5)*2+0.5

    # local search
    expected_cut = torch.zeros(graph_probs.size(dim=1))
    cnt = 0
    while True:
        cnt += 1
        for node_index in range(0, num_nodes):
            node = data.sorted_degree_nodes[node_index]
            neighbor_index = data.neighbors[node]
            neighbor_edge_weight = data.neighbor_edges[node]
            node_temp_v = torch.mm(
                neighbor_edge_weight, graph_probs[neighbor_index])
            node_temp_v = torch.squeeze(node_temp_v)
            node_temp_v += torch.rand(node_temp_v.shape[0],
                                      device=torch.device('cuda:0'))/4
            graph_probs[node] = (node_temp_v <
                                 data.weighted_degree[node]/2+0.125).int()
        if cnt >= num_ls:
            break

    expected_cut[:] = ((2 * graph_probs[nlr_graph.type(torch.long)][:] - 1)*(
        2 * graph_probs[nlc_graph.type(torch.long)][:] - 1) * edge_weight).sum(dim=0)

    expected_cut_reshape = torch.reshape(expected_cut, (-1, total_mcmc_num))
    index = torch.argmin(expected_cut_reshape, dim=0)
    for i0 in range(total_mcmc_num):
        index[i0] = i0 + index[i0]*total_mcmc_num
    max_cut = expected_cut[index]

    return ((edge_weight_sum-max_cut) / 2), graph_probs[:, index], graph_probs, start, (expected_cut-torch.mean(expected_cut)).to(device)
