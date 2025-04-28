import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, dim, inner_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, inner_dim)
        self.w2 = nn.Linear(inner_dim, dim)
        self.w3 = nn.Linear(dim, inner_dim)
    
    def forward(self, x):
        return self.w2(F.silu(self.w3(x)) * self.w1(x))

class Gate(nn.Module):
    def __init__(self, dim, num_experts, config):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.score_func = "softmax"
        self.select_expert = 8
        self.route_scale = 1.0
        self.weights = nn.Parameter(torch.randn(num_experts, dim))
        self.bias = nn.Parameter(torch.randn(num_experts))
    
    def forward(self, x):
        score = F.linear(x, self.weights)
        if self.score_func == "softmax":
            score = F.softmax(score, dim=-1)
        elif self.score_func == "sigmoid":
            score = F.sigmoid(score)

        if self.bias:
            score = score + self.bias
        original_scores = score
        indices = score.topk(self.select_expert, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        # 这里主要是因为sigmoid的和不为1，所以需要归一化
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale

        return weights.type_as(x), indices

class Moe(nn.Module):
    def __init__(self, dim, num_experts, config):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.dim = dim
        self.inner_dim = config.inner_dim
        self.select_expert = 8
        self.routed_experts = nn.ModuleList([Expert(dim, config.inner_dim) for _ in range(num_experts)])
        self.gate = Gate(dim, num_experts, config)
        self.shared_experts = Expert(dim, config.inner_dim)
    
    def forward(self, x):
        shape = x.size()
        x = x.view(-1, self.dim)
        y = torch.zeros_like(x)
        weights, indices = self.gate(x)

        counts = torch.bincount(indices.flatten(), minlength=self.num_experts).tolist()
        for i in range(len(counts)):
            if counts[i] == 0:
                continue
            expert = self.routed_experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        return (y + z).view(*shape)
    
    
        