import torch
import torch.nn as nn
from torch.nn.functional import gumbel_softmax, log_softmax
from torch.nn.utils.clip_grad import clip_grad_norm
from utils.vocab import SpecialTokens


class DecoderBase(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, max_length=20):

        super(DecoderBase, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_length = max_length

        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=SpecialTokens.pad_index)

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

    def step(self, input, state=None, memory=None):
        raise NotImplementedError

    def train_greedy_decode(self, input_sequence, initial_state=None, memory=None):
        raise NotImplementedError

    def train_gumbel_softmax_decode(self, tau, initial_state=None, memory=None):
        raise NotImplementedError

    def greedy_search(self, initial_state=None, memory=None):
        raise NotImplementedError

    def beam_search(self, beam_width, initial_state=None, memory=None):
        raise NotImplementedError


class RNNDecoder(DecoderBase):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, rnn_type="GRU", max_length=20):

        super(RNNDecoder, self).__init__(vocab_size, embedding_dim, hidden_dim, max_length=max_length)
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

        self.linear = nn.Linear(self.hidden_dim, self.vocab_size)

    def step(self, input, state=None, memory=None):

        assert state is not None

        input_size_len = len(input.size())
        assert input_size_len == 1 or input_size_len == 2
        if input_size_len == 2:
            input_emb = torch.einsum("ij,jk->ik", [input, self.token_embedding.weight])
        else:
            input_emb = self.token_embedding(input)
        input_emb = input_emb.unsqueeze(1)

        _, state = self.rnn(input_emb, state)

        logits = self.linear(state.squeeze(0))

        return logits, state

    def train_greedy_decode(self, input_sequence, initial_state=None, memory=None):

        if initial_state is None:
            state = self.initialize_state(input_sequence.size()[0])
        else:
            state = initial_state.unsqueeze(0)

        sos_ids = torch.ones_like(input_sequence[:, 0]).unsqueeze(1)
        input_sequence = torch.cat([sos_ids, input_sequence[:, :-1]], dim=-1)

        logits_record, state_record = [], []
        max_length = input_sequence.size(1)
        for step in range(max_length):
            input = input_sequence[:, step]
            logits, state = self.step(input, state=state)
            logits_record.append(logits.unsqueeze(1))
            state_record.append(state.squeeze(0).unsqueeze(1))
        logits = torch.cat(logits_record, dim=1)
        states = torch.cat(state_record, dim=1)

        return logits, states

    def train_gumbel_softmax_decode(self, tau, initial_state=None, memory=None):

        assert initial_state is not None
        state = initial_state.unsqueeze(0)
        batch_size = initial_state.size(0)
        input = torch.zeros(batch_size, self.vocab_size).float().to(self.device)
        input[:, SpecialTokens.sos_index] = 1.0

        logits_record = []
        output_record = []
        state_record = []
        for step in range(self.max_length):
            logits, state = self.step(input, state=state)
            logits_record.append(logits.unsqueeze(1))
            input = gumbel_softmax(logits, tau=tau)
            output_record.append(input.unsqueeze(1))
            state_record.append(state.squeeze(0).unsqueeze(1))

        logits = torch.cat(logits_record, dim=1)
        outputs = torch.cat(output_record, dim=1)
        states = torch.cat(state_record, dim=1)

        _, tokens = torch.max(outputs, dim=-1)
        tokens = tokens.tolist()
        lengths = []
        for i in range(len(tokens)):
            length = 0
            flag = False
            for j in range(len(tokens[i])):
                if flag:
                    outputs[i, j, :] = 0.0
                    outputs[i, j, SpecialTokens.pad_index] = 1.0
                else:
                    length += 1
                    if tokens[i][j] == SpecialTokens.eos_index:
                        flag = True
            lengths.append(length)

        return logits, outputs, states, lengths

    def initialize_state(self, batch_size):
        state = torch.zeros(1, batch_size, self.hidden_dim).to(self.device)
        return state

    def clip_grad(self):
        clip_grad_norm(self.rnn.parameters(), 5.0)

    def greedy_search(self, initial_state=None, memory=None):

        assert initial_state is not None
        batch_size = initial_state.size(0)
        state = initial_state.unsqueeze(0)

        input = torch.ones(batch_size).long().to(self.device) * SpecialTokens.sos_index
        record = []
        for step in range(self.max_length):
            logits, state = self.step(input, state)
            _, input = torch.max(logits, dim=-1)
            record.append(input.unsqueeze(1))
        outputs = torch.cat(record, dim=-1)
        outputs = outputs.tolist()

        return outputs

    def beam_search(self, beam_width, initial_state=None, memory=None):

        assert initial_state is not None
        batch_size = initial_state.size(0)
        outputs = []
        for i in range(batch_size):
            input = torch.ones(1).long().to(self.device) * SpecialTokens.sos_index
            state = initial_state[i].unsqueeze(0).unsqueeze(0)
            beam = [[[], 0, input, state.clone(), False]]  # [output, log_prob, input, state, is_end]
            for _ in range(self.max_length):
                new_beam = []
                for i in range(len(beam)):
                    item = beam[i]
                    if len(item[0]) >= self.max_length or item[4]:
                        new_beam.append(item)
                    else:
                        [_, _, input, state, _] = item
                        logits, state = self.step(input, state)
                        logits = log_softmax(logits, dim=-1)
                        topk_logits, topk_tokens = torch.topk(logits, beam_width, dim=-1)
                        topk_logits = topk_logits.squeeze(0).tolist()
                        topk_tokens = topk_tokens.squeeze(0).tolist()
                        for log_prob, token in zip(topk_logits, topk_tokens):
                            new_item = [item[0] + [token],
                                        (item[1] * len(item[0]) - log_prob) / (len(item[0]) + 1),
                                        torch.ones_like(input) * token,
                                        state,
                                        token == SpecialTokens.eos_index]
                            new_beam.append(new_item)
                new_beam.sort(key=lambda item: item[1])
                beam = new_beam[0:beam_width]
            outputs.append(beam[0][0])

        return outputs
