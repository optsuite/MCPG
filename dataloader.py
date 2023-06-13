import torch
from torch_geometric.data import Data
import scipy.io as scio
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dataloader_selector(problem_type):
    if problem_type == "maxcut":
        return dataloader_maxcut
    elif problem_type == "maxsat":
        return dataloader_maxsat
    elif problem_type == "MIMO":
        return dataloader_mimo
    elif problem_type == "cheeger cut":
        return dataloader_maxcut
    else:
        raise(Exception("Unrecognized problem type {}".format(problem_type)))


def dataloader_maxcut(path):
    with open(path) as f:
        fline = f.readline()
        fline = fline.split()
        num_nodes, num_edges = int(fline[0]), int(fline[1])
        edge_index = torch.LongTensor(2, num_edges)
        edge_attr = torch.Tensor(num_edges, 1)
        cnt = 0
        while True:
            lines = f.readlines(num_edges * 2)
            if not lines:
                break
            for line in lines:
                line = line.rstrip('\n').split()
                edge_index[0][cnt] = int(line[0]) - 1
                edge_index[1][cnt] = int(line[1]) - 1
                edge_attr[cnt][0] = float(line[2])
                cnt += 1
        data_Gset = Data(num_nodes=num_nodes,
                         edge_index=edge_index, edge_attr=edge_attr)
        data_Gset = data_Gset.to(device)
        data_Gset.edge_weight_sum = float(torch.sum(data_Gset.edge_attr))

        data_Gset = append_neighbors_maxcut(data_Gset)

        data_Gset.single_degree = []
        data_Gset.weighted_degree = []
        tensor_abs_weighted_degree = []
        for i0 in range(data_Gset.num_nodes):
            data_Gset.single_degree.append(len(data_Gset.neighbors[i0]))
            data_Gset.weighted_degree.append(
                float(torch.sum(data_Gset.neighbor_edges[i0])))
            tensor_abs_weighted_degree.append(
                float(torch.sum(torch.abs(data_Gset.neighbor_edges[i0]))))
        tensor_abs_weighted_degree = torch.tensor(tensor_abs_weighted_degree)
        data_Gset.sorted_degree_nodes = torch.argsort(
            tensor_abs_weighted_degree, descending=True)

        edge_degree = []
        add = torch.zeros(3, num_edges).to(device)
        for i0 in range(num_edges):
            edge_degree.append(abs(edge_attr[i0].item())*(
                tensor_abs_weighted_degree[edge_index[0][i0]]+tensor_abs_weighted_degree[edge_index[1][i0]]))
            node_r = edge_index[0][i0]
            node_c = edge_index[1][i0]
            add[0][i0] = - data_Gset.weighted_degree[node_r] / \
                2 + data_Gset.edge_attr[i0] - 0.05
            add[1][i0] = - data_Gset.weighted_degree[node_c] / \
                2 + data_Gset.edge_attr[i0] - 0.05
            add[2][i0] = data_Gset.edge_attr[i0]+0.05

        for i0 in range(num_nodes):
            data_Gset.neighbor_edges[i0] = data_Gset.neighbor_edges[i0].unsqueeze(
                0)
        data_Gset.add = add
        edge_degree = torch.tensor(edge_degree)
        data_Gset.sorted_degree_edges = torch.argsort(
            edge_degree, descending=True)

        print("Data Summary: ", data_Gset)

        return data_Gset


def append_neighbors_maxcut(data):
    data.neighbors = []
    data.neighbor_edges = []
    num_nodes = data.num_nodes
    for i in range(num_nodes):
        data.neighbors.append([])
        data.neighbor_edges.append([])
    edge_number = data.edge_index.shape[1]
    print(edge_number)

    for index in range(0, edge_number):
        row = data.edge_index[0][index]
        col = data.edge_index[1][index]
        edge_weight = data.edge_attr[index][0].item()

        data.neighbors[row].append(col.item())
        data.neighbor_edges[row].append(edge_weight)
        data.neighbors[col].append(row.item())
        data.neighbor_edges[col].append(edge_weight)

    data.n0 = []
    data.n1 = []
    data.n0_edges = []
    data.n1_edges = []
    for index in range(0, edge_number):
        row = data.edge_index[0][index]
        col = data.edge_index[1][index]
        data.n0.append(data.neighbors[row].copy())
        data.n1.append(data.neighbors[col].copy())
        data.n0_edges.append(data.neighbor_edges[row].copy())
        data.n1_edges.append(data.neighbor_edges[col].copy())
        i = 0
        for i in range(len(data.n0[index])):
            if data.n0[index][i] == col:
                break
        data.n0[index].pop(i)
        data.n0_edges[index].pop(i)
        for i in range(len(data.n1[index])):
            if data.n1[index][i] == row:
                break
        data.n1[index].pop(i)
        data.n1_edges[index].pop(i)

        data.n0[index] = torch.LongTensor(data.n0[index]).to(device)
        data.n1[index] = torch.LongTensor(data.n1[index]).to(device)
        data.n0_edges[index] = torch.tensor(
            data.n0_edges[index]).unsqueeze(0).to(device)
        data.n1_edges[index] = torch.tensor(
            data.n1_edges[index]).unsqueeze(0).to(device)
    for i in range(num_nodes):
        data.neighbors[i] = torch.LongTensor(data.neighbors[i]).to(device)
        data.neighbor_edges[i] = torch.tensor(
            data.neighbor_edges[i]).to(device)

    return data


def dataloader_maxsat(f, ptype):
    lines = f.readlines()
    variable_index = []
    clause_index = []
    neg_index = []
    clause_cnt = 0
    nhard = 0
    nvi = []
    nci = []
    nneg = []
    tempvi = []
    tempneg = []
    vp = []
    vn = []
    for line in lines:
        line = line.split()
        if len(line) == 0:
            continue
        elif line[0] == "c":
            continue
        elif line[0] == "p":
            if ptype == 'p':
                weight = int(line[4])
            nvar, nclause = int(line[2]), int(line[3])
            for i0 in range(nvar):
                nvi.append([])
                nci.append([])
                nneg.append([])
            vp = [0]*nvar
            vn = [0]*nvar
            continue
        tempvi = []
        tempneg = []
        if ptype == 'p':
            clause_weight_i = int(line[0])
            if clause_weight_i == weight:
                nhard += 1
            for ety in line[1:-1]:
                ety = int(ety)
                variable_index.append(abs(ety) - 1)
                tempvi.append(abs(ety) - 1)
                clause_index.append(clause_cnt)
                neg_index.append(int(ety/abs(ety))*clause_weight_i)
                tempneg.append(int(ety/abs(ety))*clause_weight_i)
                if ety > 0:
                    vp[abs(ety) - 1] += 1
                else:
                    vn[abs(ety) - 1] += 1
        else:
            for ety in line:
                if ety == '0':
                    continue
                ety = int(ety)
                variable_index.append(abs(ety) - 1)
                tempvi.append(abs(ety) - 1)
                clause_index.append(clause_cnt)
                neg_index.append(int(ety/abs(ety)))
                tempneg.append(int(ety/abs(ety)))
                if ety > 0:
                    vp[abs(ety) - 1] += 1
                else:
                    vn[abs(ety) - 1] += 1
        for i0 in range(len(tempvi)):
            node = tempvi[i0]
            nvi[node] += tempvi
            nneg[node] += tempneg
            temp = len(nci[node])
            if temp > 0:
                temp = nci[node][temp-1]+1
            nci[node] += [temp]*len(tempvi)
        clause_cnt += 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    degree = []
    for i0 in range(nvar):
        nvi[i0] = torch.LongTensor(nvi[i0]).to(device)
        nci[i0] = torch.LongTensor(nci[i0]).to(device)
        nneg[i0] = torch.tensor(nneg[i0]).to(device)
        degree.append(vp[i0]+vn[i0])
    degree = torch.FloatTensor(degree).to(device)
    sorted = torch.argsort(degree, descending=True).to('cpu')
    neg_index = torch.tensor(neg_index).to(device)
    ci_cuda = torch.tensor(clause_index).to(device)
    ndata = [nvi, nci, nneg, sorted, degree]
    pdata = [nvar, nclause, variable_index, ci_cuda, neg_index]
    if ptype == 'p':
        pdata = [nvar, nclause, variable_index,
                 ci_cuda, neg_index, weight, nhard]
    return pdata, ndata


def sort_node(ndata):
    degree = ndata[4]
    device = degree.device
    temp = degree + (torch.rand(degree.shape[0], device=device)-0.5)/2
    sorted = torch.argsort(temp, descending=True).to('cpu')
    ndata[3] = sorted
    return ndata
