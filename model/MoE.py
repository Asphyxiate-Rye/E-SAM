import math
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import math
import torch.nn.functional as F
import torchvision

class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)



class ConvExpert(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),#groups = in_channels
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.net(x)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class TransConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
        nn.ConvTranspose2d(in_channels, in_channels // 4, kernel_size=2,
                           stride=2),
        LayerNorm2d(in_channels // 4),
        nn.GELU(),
        nn.ConvTranspose2d(in_channels // 4, out_channels, kernel_size=2,
                           stride=2), 
        LayerNorm2d(out_channels),
        nn.GELU(),)

    def forward(self, x):
        return self.net(x)

class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.linear(mh_output)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class NoisyTopkRouter_cv(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter_cv, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output, mh_output_avg):
        logits = self.topkroute_linear(mh_output_avg)
        noise_logits = self.noise_linear(mh_output)

        noise = torch.randn_like(logits) * F.softplus(noise_logits).mean(dim=1)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class ExpertChoiceTokenSparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(ExpertChoiceTokenSparseMoE, self).__init__()
        self.router = ExpertChoiceTokenNoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        bs, seq_len, dim = x.size()
        gating_output, indices = self.router(x)
        flat_x = x.view(-1, x.size(-1))
        final_output = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            x_ = flat_x[indices[i]]
            expert_output = expert(x_)

            gating_scores = gating_output[i, indices[i]].unsqueeze(1)
            weighted_output = expert_output * gating_scores

            final_output[indices[i]] += weighted_output.squeeze(1)

        return final_output.reshape(bs, seq_len, dim) + x, indices


class ExpertChoiceTokenNoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(ExpertChoiceTokenNoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output).reshape(-1, self.num_experts).T
        noise_logits = self.noise_linear(mh_output).reshape(-1, self.num_experts).T

        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class SparseMoE_cv(nn.Module):
    def __init__(self, n_embed, out_channels, num_experts, top_k=2):
        super(SparseMoE_cv, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.out_channels = out_channels
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.avgpool_layers = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        bs, dim, h, w = x.shape
        final_output = torch.zeros([bs, self.out_channels, h*4, w*4]).cuda()

        x_avg = self.avgpool_layers(x)
        x_avg = x_avg.reshape(bs, dim)
        x = x.reshape(bs, h*w, dim)
        gating_output, indices = self.router(x, x_avg)     # [b, h*w, num_experts]     [b, h*w, topk] --> []

        flat_x = x.reshape(bs, dim, h, w)      # [b*h*w, dim]
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))   # [b*h*w, num_experts]

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)   # [b, h*w]  -->  [num_features]
            flat_mask = expert_mask.view(-1)           # [b*h*w]  -->   [num_features]

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
                weighted_output = expert_output * gating_scores

                final_output[expert_mask] += weighted_output

        return final_output


class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        # layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output




class TransormerBlock(nn.Module):
    """ Mixture of Experts Transformer block: communication followed by computation (multi-head self attention + SparseMoE) """

    def __init__(self, n_embed, n_head, num_experts, top_k):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.smoe(self.ln2(x))
        return x


def kaiming_init_weights(m):
    if isinstance (m, (nn.Linear)):
        init.kaiming_normal_(m.weight)


if __name__ == "__main__":
    model = SparseMoE(48, 4, 2).cuda()

    a = torch.randn(4, 64, 64, 16, 48).cuda()
    b, h, w, d, dim = a.shape
    a = a.view(b, h*w*d, dim)
    out = model(a)
    print(out.shape)
