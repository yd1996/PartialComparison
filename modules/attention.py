import torch
import torch.nn as nn


class AttentionBase(nn.Module):

    def __init__(self, dim):

        super(AttentionBase, self).__init__()
        self.dim = dim

    def calcuate_attention_weight(self, query, key):
        raise NotImplementedError

    def forward(self, query, memory):
        attention_weight = self.calcuate_attention_weight(query, memory)
        weighted_memory = torch.einsum("ijk,ij->ik", [memory, attention_weight])
        return weighted_memory