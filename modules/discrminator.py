import torch
import torch.nn as nn
from torch.nn.functional import tanh, softmax


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class MLPDiscriminator(nn.Module):

    def __init__(self, dims, label_size):

        super(MLPDiscriminator, self).__init__()
        assert len(dims) == 3
        self.dims = dims
        self.label_size = label_size
        self.layer1 = nn.Linear(dims[0], dims[1])
        self.layer2 = nn.Linear(dims[1], dims[2])
        self.layer3 = nn.Linear(dims[2], label_size)

    def forward(self, input):

        logits = self.layer1(input)
        logits = tanh(logits)
        logits = self.layer2(logits)
        logits = tanh(logits)
        logits = self.layer3(logits)
        probs = softmax(logits, dim=-1)

        return logits, probs


class CNNDiscriminator(nn.Module):

    def __init__(self, max_length, input_dim, kernel_sizes, label_size=2):

        super(CNNDiscriminator, self).__init__()
        self.max_length = max_length
        self.kernel_sizes = kernel_sizes

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(self.kernel_sizes[0], input_dim)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(self.kernel_sizes[0], 1)))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(self.kernel_sizes[1], input_dim)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(self.kernel_sizes[1], 1)))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(self.kernel_sizes[2], input_dim)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(self.kernel_sizes[2], 1)))

        out_dim = 0
        for i in range(3):
            dim = self.max_length - self.kernel_sizes[i] + 1
            dim = (dim - self.kernel_sizes[i]) / self.kernel_sizes[i] + 1
            dim = int(dim)
            out_dim += dim

        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, label_size))

    def forward(self, sentences):
        features = self.encode(sentences)
        logits = self.mlp(features)
        probs = softmax(logits)
        return logits, probs

    def encode(self, sentences_emb):

        if sentences_emb.size(1) > self.max_length:
            sentences_emb = sentences_emb[:, :self.max_length, :]
        elif sentences_emb.size(1) < self.max_length:
            batch_size = sentences_emb.size(0)
            length = sentences_emb.size(1)
            dim = sentences_emb.size(2)
            padded_zeros = torch.ones(batch_size, self.max_length - length, dim).float().to(device)
            sentences_emb = torch.cat([sentences_emb, padded_zeros], dim=1)
        sentences_emb = sentences_emb.unsqueeze(1)
        feature_map_0 = self.conv0(sentences_emb).squeeze(3)
        feature_map_1 = self.conv1(sentences_emb).squeeze(3)
        feature_map_2 = self.conv2(sentences_emb).squeeze(3)
        feature_map = torch.cat((feature_map_0, feature_map_1, feature_map_2), dim=2)
        feature_map = feature_map.view(sentences_emb.size(0), -1)

        return feature_map