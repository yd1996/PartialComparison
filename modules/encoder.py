import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
from utils.vocab import SpecialTokens


class EncoderBase(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, max_length=20):

        super(EncoderBase, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_length = max_length

        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=SpecialTokens.pad_index)
        self.token_embedding_dropout = nn.Dropout(p=0.1)

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

    def forward(self, input_sequence, input_length):
        raise NotImplementedError


class RNNEncoder(EncoderBase):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, rnn_type="GRU"):

        super(RNNEncoder, self).__init__(vocab_size, embedding_dim, hidden_dim)
        self.rnn_type = rnn_type
        if self.rnn_type == "GRU":
            self.rnn_constructor = nn.GRU
        elif self.rnn_type == "LSTM":
            self.rnn_constructor = nn.LSTM
        else:
            self.rnn_constructor = nn.RNN

        self.rnn = self.rnn_constructor(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1)

    def forward(self, input_sequence, input_length):

        input_size_len = len(input_sequence.size())
        assert input_size_len == 2 or input_size_len == 3
        if input_size_len == 3:
            input_emb = torch.einsum("ijk,kl->ijl", [input_sequence, self.token_embedding.weight])
        else:
            input_emb = self.token_embedding(input_sequence)
        input_emb = self.token_embedding_dropout(input_emb)

        rnn_outputs, _ = self.rnn(input_emb)
        if input_size_len == 3:
            mask = torch.zeros_like(input_sequence[:, :, 0]).byte()
        else:
            mask = torch.zeros_like(input_sequence).byte()
        for i in range(len(input_length)):
            j = input_length[i] - 1
            mask[i, j] = 1
        final_state = rnn_outputs[mask]

        return rnn_outputs, final_state

    def inference(self, input_sequence, input_length):

        input_size_len = len(input_sequence.size())
        assert input_size_len == 2 or input_size_len == 3
        if input_size_len == 3:
            input_emb = torch.einsum("ijk,kl->ijl", [input_sequence, self.token_embedding.weight])
        else:
            input_emb = self.token_embedding(input_sequence)

        rnn_outputs, _ = self.rnn(input_emb)

        mask = torch.zeros_like(input_sequence).byte()
        for i in range(len(input_length)):
            j = input_length[i] - 1
            mask[i, j] = 1
        final_state = rnn_outputs[mask]

        return rnn_outputs, final_state

    def clip_grad(self):
        clip_grad_norm(self.rnn.parameters(), 5.0)

