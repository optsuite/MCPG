import torch


class simple(torch.nn.Module):
    def __init__(self, hidden_1, hidden_2, output_num, deltas):
        super(simple, self).__init__()
        self.lin = torch.nn.Linear(1,  output_num)
        self.sigmoid = torch.nn.Sigmoid()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, data, alpha=0.1, start_samples=None, value=None, device=torch.device("cuda")):
        x = torch.ones(1).to(device)
        x = self.lin(x)
        x = self.sigmoid(x)

        x = (x-0.5) * 0.6 + 0.5
        probs = x
        probs = probs.squeeze()
        retdict = {}
        reg = probs * torch.log(probs) + (1-probs) * torch.log(1-probs)
        reg = torch.mean(reg)
        if start_samples == None:
            retdict["output"] = [probs.squeeze(-1), "hist"]  # output
            retdict["reg"] = [reg, "sequence"]
            retdict["loss"] = [alpha * reg, "sequence"]
            return retdict

        # calculate the cut for samples
        res_samples = value.t().detach()

        start_samples_idx = start_samples * \
            probs + (1 - start_samples) * (1 - probs)
        log_start_samples_idx = torch.log(start_samples_idx)
        log_start_samples = log_start_samples_idx.sum(dim=1)
        loss_ls = torch.mean(log_start_samples * res_samples)

        loss = loss_ls + alpha * reg

        retdict["output"] = [probs.squeeze(-1), "hist"]  # output
        retdict["reg"] = [reg, "sequence"]
        retdict["loss"] = [loss, "sequence"]
        return retdict

    def __repr__(self):
        return self.__class__.__name__
