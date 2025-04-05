# %%

import torch.cuda.amp as amp
from transformers.models.gpt2.modeling_gpt2 import Conv1D
import torch 
import torch.nn as nn 
import math 

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer to inject trainable parameters A and B into original weight update.
    """
    def __init__(self, in_dim, out_dim, rank, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # Low-rank matrices
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.empty(rank, out_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # Explanation log: GPT-2 base has hidden_dim=768; rank=16, alpha=32 by default.
        print(f"[LoRALayer] in_dim={in_dim}, out_dim={out_dim}, rank={rank}, alpha={alpha}")

    def forward(self, x):
        # Decomposition: alpha * (x @ A @ B)
        return self.alpha * (x @ self.A @ self.B)

class LinearWithLoRA(nn.Module):
    """
    Wrapper for nn.Linear that adds a LoRA output to the original linear output.
    """
    def __init__(self, linear_module, rank, alpha=1.0):
        super().__init__()
        self.linear = linear_module
        self.lora   = LoRALayer(linear_module.in_features, linear_module.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

class Conv1DWithLoRA(nn.Module):
    """
    Wrapper for Conv1D that adds a LoRA output to the original Conv1D output.
    """
    def __init__(self, conv1d_module: Conv1D, rank, alpha=1.0):
        super().__init__()
        self.conv = conv1d_module
        in_dim, out_dim = conv1d_module.weight.shape
        self.lora = LoRALayer(in_dim, out_dim, rank, alpha)

    def forward(self, x):
        out_normal = self.conv(x)
        B, S, hidden_dim = x.shape
        x_2d = x.view(B*S, hidden_dim)
        out_lora_2d = self.lora(x_2d)
        out_lora_3d = out_lora_2d.view(B, S, -1)
        return out_normal + out_lora_3d

def replace_modules_with_lora(module, rank=16, alpha=32):
    """
    Recursively replace GPT-2 submodules (c_fc, c_proj) with LoRA wrappers.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in ["c_fc", "c_proj"]:
            new_module = LinearWithLoRA(child, rank, alpha)
            setattr(module, name, new_module)
        elif isinstance(child, Conv1D) and name in ["c_fc", "c_proj"]:
            new_module = Conv1DWithLoRA(child, rank, alpha)
            setattr(module, name, new_module)
        else:
            replace_modules_with_lora(child, rank, alpha)

def freeze_original_parameters(model):
    """
    Freeze all parameters except LoRA layers and classifier.
    """
    for name, param in model.named_parameters():
        if "lora" not in name.lower() and "classifier" not in name.lower():
            param.requires_grad = False

def print_trainable_parameters(model):
    """
    Print the number of trainable parameters vs. total parameters.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / Total params: {total}")
    return trainable

def show_gradient_norms(model):
    """
    Print gradient norms for LoRA layers to confirm only LoRA + classifier receive gradients.
    """
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            print(f"Gradient Norm for {name}: {param.grad.norm():.4f}")