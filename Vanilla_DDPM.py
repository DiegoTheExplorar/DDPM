import torch
import torch.nn.functional as F
import math

def cosine_beta_schedule(timesteps, start=0.0001, end=0.02):
    return start + 0.5 * (end - start) * (1 + torch.cos(torch.linspace(0, math.pi, timesteps)))