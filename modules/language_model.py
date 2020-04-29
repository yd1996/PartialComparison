import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.functional import log_softmax
from torch.optim import Adam

from tensorboardX import SummaryWriter

from .loss import calcuate_maximum_likehood_loss, calculate_perplexity
from utils.vocab import SpecialTokens
from utils.dataloader import get_sample_iterator, get_eval_sample_iterator


class LanguageModel(nn.Module):

    def __init__(self, vocab_size, dim):

        super(LanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=SpecialTokens.pad_index)
        self.rnn = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1,
                           batch_first=True, bidirectional=False)
        self.linear = nn.Linear(dim, vocab_size)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def train_maximum_likehood(self, input_seq):

        sos_ids = torch.ones_like(input_seq[:, 0]).unsqueeze(1).to(self.device)
        rnn_input = torch.cat([sos_ids, input_seq[:, :-1]], dim=1)
        input_emb = self.embedding(rnn_input)
        batch_size = input_seq.size(0)
        state = tuple([torch.zeros(1, batch_size, self.dim).float().to(self.device) for _ in range(2)])
        rnn_outputs, _ = self.rnn(input_emb, state)
        rnn_outputs = self.linear(rnn_outputs)
        loss_mle, loss_mle_val = calcuate_maximum_likehood_loss(rnn_outputs, input_seq, pad_id=SpecialTokens.pad_index)

        return loss_mle, loss_mle_val

    def calculate_perplexity(self, input_seq):

        sos_ids = torch.ones_like(input_seq[:, 0]).unsqueeze(1).to(self.device)
        rnn_input = torch.cat([sos_ids, input_seq[:, :-1]], dim=1)
        input_emb = self.embedding(rnn_input)
        batch_size = input_seq.size(0)
        state = tuple([torch.zeros(1, batch_size, self.dim).float().to(self.device) for _ in range(2)])
        rnn_outputs, _ = self.rnn(input_emb, state)
        rnn_outputs = self.linear(rnn_outputs)
        perplexity = calculate_perplexity(rnn_outputs, input_seq, pad_id=SpecialTokens.pad_index)
        perplexity = perplexity.tolist()

        return perplexity

    def calculate_reward_signal(self, input_seq):

        sos_ids = torch.zeros_like(input_seq[:, 0, :]).unsqueeze(1)
        sos_ids[:, :, SpecialTokens.sos_index] = 1.0
        rnn_input = torch.cat([sos_ids, input_seq[:, :-1, :]], dim=1)
        input_emb = torch.einsum("ijk,kl->ijl", [rnn_input, self.embedding.weight])

        batch_size = input_seq.size(0)
        state = tuple([torch.zeros(1, batch_size, self.dim).float().to(self.device) for _ in range(2)])
        rnn_outputs, _ = self.rnn(input_emb, state)
        rnn_outputs = self.linear(rnn_outputs)
        predictions = log_softmax(rnn_outputs)

        return predictions

    def clip_grad(self):
        clip_grad_norm(self.rnn.parameters(), 1.0)


hidden_dim = 512
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def build_language_model(vocab, checkpoint_path=None):
    language_model = LanguageModel(vocab.get_size(), hidden_dim).to(device)
    if checkpoint_path is not None:
        language_model.load_state_dict(torch.load(checkpoint_path))
    return language_model


def train_language_model(train_paths, valid_paths, vocab, checkpoint_dir):

    language_model = LanguageModel(vocab.get_size(), hidden_dim).to(device)
    optimizer = Adam(params=language_model.parameters(), lr=1e-4)

    train_iterators = [get_sample_iterator(path, 50, 100000, shuffle=True) for path in train_paths]
    summary_writer = SummaryWriter(log_dir=checkpoint_dir)

    step = 0
    while step < 100000:

        texts = []
        for train_iter in train_iterators:
            texts_tmp, step = next(train_iter)
            texts = texts + texts_tmp
        text_ids, _ = vocab.encode_sequence_batch(texts)
        text_ids = torch.tensor(text_ids).long().to(device)

        language_model.train()
        optimizer.zero_grad()
        loss_mle, loss_mle_val = language_model.train_maximum_likehood(text_ids)
        loss_mle.backward()
        language_model.clip_grad()
        optimizer.step()

        print("[train] step=%d loss=%.5f" % (step, loss_mle_val.tolist()))
        summary_writer.add_scalar("loss", loss_mle_val.tolist(), global_step=step)

        if step % 500 == 0:
            perplexity_list = []
            language_model.eval()
            valid_iterators = [get_eval_sample_iterator(path, 50) for path in valid_paths]
            for valid_iter in valid_iterators:
                for texts in valid_iter:
                    text_ids, _ = vocab.encode_sequence_batch(texts)
                    text_ids = torch.tensor(text_ids).long().to(device)
                    perplexity = language_model.calculate_perplexity(text_ids)
                    perplexity_list = perplexity_list + perplexity
            perplexity_val = sum(perplexity_list) / len(perplexity_list)
            print("[eval] step=%d ppl=%.5f" % (step, perplexity_val))
            summary_writer.add_scalar("ppl", perplexity_val, global_step=step)

        if step > 0 and step % 5000 == 0:
            torch.save(language_model.state_dict(), checkpoint_dir + "checkpoint.%d" % step)

    summary_writer.close()


def evaluate(paths, vocab, checkpoint_path):

    language_model = LanguageModel(vocab.get_size(), hidden_dim).to(device)
    language_model.load_state_dict(torch.load(checkpoint_path))

    perplexity_list = []
    language_model.eval()
    valid_iterators = [get_eval_sample_iterator(path, 50) for path in paths]
    for valid_iter in valid_iterators:
        for texts in valid_iter:
            text_ids, _ = vocab.encode_sequence_batch(texts)
            text_ids = torch.tensor(text_ids).long().to(device)
            perplexity = language_model.calculate_perplexity(text_ids)
            perplexity_list = perplexity_list + perplexity
    perplexity_val = sum(perplexity_list) / len(perplexity_list)
    print("[eval] ppl=%.5f" % perplexity_val)
