import torch
import torch.nn as nn
from utils.vocab import SpecialTokens


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class ContentComparator(nn.Module):

    def __init__(self, vocab_size, embedding_dim, max_length, kernel_sizes):

        super(ContentComparator, self).__init__()
        self.max_length = max_length + 1
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=SpecialTokens.pad_index)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.kernel_sizes[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.kernel_sizes[0], stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.kernel_sizes[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.kernel_sizes[1], stride=1))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.kernel_sizes[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.kernel_sizes[2], stride=1))

        dim = self.max_length
        for i in range(len(self.kernel_sizes)):
            for _ in range(2):
                dim = dim - self.kernel_sizes[i] + 1
        out_dim = dim * dim

        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, 2))

    def forward(self, src_sentences, tgt_sentences):
        src_sent_emb = self.encode(src_sentences)
        tgt_sent_emb = self.encode(tgt_sentences)
        tgt_sent_emb = tgt_sent_emb.transpose(1, 2)
        embedding_matrix = torch.bmm(src_sent_emb, tgt_sent_emb)
        embedding_matrix = embedding_matrix.unsqueeze(1)
        feature_matrix = self.layer1(embedding_matrix)
        feature_matrix = self.layer2(feature_matrix)
        feature_matrix = self.layer3(feature_matrix)
        batch_size = src_sentences.size(0)
        features = feature_matrix.view(batch_size, -1)
        logits = self.mlp(features)
        return logits

    def encode(self, sentences):
        if len(sentences.size()) == 3:
            sent_emb = torch.einsum("ijk,kl->ijl", [sentences, self.embedding.weight])
        else:
            sent_emb = self.embedding(sentences)
        if sent_emb.size(1) > self.max_length:
            sent_emb = sent_emb[:, :self.max_length, :]
        elif sent_emb.size(1) < self.max_length:
            padded_zeros = torch.zeros(sent_emb.size(0), self.max_length - sent_emb.size(1),
                                       sent_emb.size(2)).float().to(device)
            sent_emb = torch.cat([sent_emb, padded_zeros], dim=1)
        return sent_emb


class StyleComparator(nn.Module):

    def __init__(self, vocab_size, embedding_dim, max_length, kernel_sizes):

        super(StyleComparator, self).__init__()
        self.max_length = max_length + 1
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=SpecialTokens.pad_index)
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(self.kernel_sizes[0], embedding_dim)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(self.kernel_sizes[0], 1)))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(self.kernel_sizes[1], embedding_dim)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(self.kernel_sizes[1], 1)))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(self.kernel_sizes[2], embedding_dim)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(self.kernel_sizes[2], 1)))

        out_dim = 0
        for i in range(3):
            dim = self.max_length - self.kernel_sizes[i] + 1
            dim = (dim - self.kernel_sizes[i]) / self.kernel_sizes[i] + 1
            dim = int(dim)
            out_dim += dim
        out_dim *= 2

        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, 2))

    def forward(self, src_sentences, tgt_sentences):
        src_features = self.encode(src_sentences)
        tgt_features = self.encode(tgt_sentences)
        features = torch.cat((src_features, tgt_features), dim=1)
        logits = self.mlp(features)
        return logits

    def encode(self, sentences):
        if len(sentences.size()) == 3:
            sentences_emb = torch.einsum("ijk,kl->ijl", [sentences, self.embedding.weight])
        else:
            sentences_emb = self.embedding(sentences)
        if sentences_emb.size(1) > self.max_length:
            sentences_emb = sentences_emb[:, :sentences_emb.size(1), :]
        elif sentences_emb.size(1) < self.max_length:
            padded_zeros = torch.zeros(sentences_emb.size(0), self.max_length - sentences_emb.size(1),
                                       sentences_emb.size(2)).float().to(device)
            sentences_emb = torch.cat([sentences_emb, padded_zeros], dim=1)
        sentences_emb = sentences_emb.unsqueeze(1)
        feature_map_0 = self.conv0(sentences_emb).squeeze(3)
        feature_map_1 = self.conv1(sentences_emb).squeeze(3)
        feature_map_2 = self.conv2(sentences_emb).squeeze(3)
        feature_map = torch.cat((feature_map_0, feature_map_1, feature_map_2), dim=2)
        feature_map = feature_map.view(sentences.size(0), -1)
        return feature_map
