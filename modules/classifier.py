import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.optim import Adam
from tensorboardX import SummaryWriter
from utils.dataloader import get_sample_iterator, get_eval_sample_iterator
from utils.vocab import SpecialTokens
from modules.discrminator import CNNDiscriminator


class AttentiveRNNClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, label_size):

        super(AttentiveRNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
            dropout=0.1)

        self.linear_w = nn.Linear(hidden_dim * 2, hidden_dim)
        self.tanh = nn.Tanh()
        self.linear_a = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

        self.linear = nn.Linear(hidden_dim * 2, label_size)

    def forward(self, input_seq, input_len):

        input_emb = self.embedding(input_seq)
        rnn_outputs, _ = self.rnn(input_emb)
        u = self.linear_w(rnn_outputs)
        u = self.tanh(u)
        u = self.linear_a(u).squeeze(-1)
        mask = torch.zeros_like(u)
        for i in range(len(input_len)):
            for j in range(input_len[i], mask.size(1)):
                mask[i, j] = -10000
        u = u + mask
        attention_weight = self.softmax(u)
        encoding = torch.einsum("ijk,ij->ik", [rnn_outputs, attention_weight])
        logits = self.linear(encoding)
        probs = self.softmax(logits)

        return logits, probs

    def clip_grad(self):
        clip_grad_norm(self.rnn.parameters(), 5.0)


class CNNClassifier(nn.Module):

    def __init__(self, vocab_size, max_length, embedding_dim, kernel_sizes):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=SpecialTokens.pad_index)
        self.cnn_discriminator = CNNDiscriminator(max_length, embedding_dim, kernel_sizes)

    def forward(self, input_seq):
        if len(input_seq.size()) == 2:
            input_emb = self.embedding(input_seq)
        else:
            input_emb = torch.einsum("ijk,kl->ijl", [input_seq, self.embedding.weight])
        logits, probs = self.cnn_discriminator.forward(input_emb)
        return logits, probs


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def train_sentence_classifier(train_paths, valid_paths, vocab, max_length, checkpoint_dir, type="cnn"):

    if type == "cnn":
        sentence_classifier = CNNClassifier(vocab.get_size(), max_length, 300, [3, 4, 5]).to(device)
    else:
        sentence_classifier = AttentiveRNNClassifier(vocab.get_size(), 300, 300, 2).to(device)
    optimizer = Adam(params=sentence_classifier.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()

    train_iterators = [get_sample_iterator(path, 50, 20000, shuffle=True) for path in train_paths]
    summary_writer = SummaryWriter(log_dir=checkpoint_dir)

    step = 0
    while step < 20000:

        texts = []
        labels = []
        for i in range(len(train_iterators)):
            texts_tmp, step = next(train_iterators[i])
            texts = texts + texts_tmp
            labels = labels + [i] * len(texts_tmp)
        text_ids, lengths = vocab.encode_sequence_batch(texts)
        text_ids = torch.tensor(text_ids).long().to(device)
        labels = torch.tensor(labels).long().to(device)

        sentence_classifier.train()
        if type == "cnn":
            logits, _ = sentence_classifier.forward(text_ids)
        else:
            logits, _ = sentence_classifier.forward(text_ids, lengths)
        loss = loss_func(logits, labels)
        loss.backward()
        if type != "cnn":
            sentence_classifier.clip_grad()
        optimizer.step()

        print("[train] step=%d loss=%.5f" % (step, loss.tolist()))
        summary_writer.add_scalar("loss", loss.tolist(), global_step=step)

        if step % 500 == 0:
            total_count, right_count = 0, 0
            sentence_classifier.eval()
            valid_iterators = [get_eval_sample_iterator(path, 50) for path in valid_paths]
            for i in range(len(valid_iterators)):
                for texts in valid_iterators[i]:
                    labels = [i] * len(texts)
                    total_count += len(labels)
                    labels = torch.tensor(labels).long().to(device)
                    text_ids, lengths = vocab.encode_sequence_batch(texts)
                    text_ids = torch.tensor(text_ids).long().to(device)
                    if type == "cnn":
                        logits, probs = sentence_classifier.forward(text_ids)
                    else:
                        logits, probs = sentence_classifier.forward(text_ids, lengths)
                    _, pred_labels = torch.max(probs, dim=1)
                    right_count += torch.sum((labels == pred_labels).float()).tolist()
            print("[eval] step=%d acc=%.5f" % (step, right_count / total_count))
            summary_writer.add_scalar("accuracy", right_count / total_count, global_step=step)

        if step > 0 and step % 5000 == 0:
            torch.save(sentence_classifier.state_dict(), checkpoint_dir + "checkpoint.%d" % step)


def evaluate(paths, vocab, max_length, checkpoint_path, type="cnn"):

    if type == "cnn":
        sentence_classifier = CNNClassifier(vocab.get_size(), max_length, 300, [3, 4, 5]).to(device)
    else:
        sentence_classifier = AttentiveRNNClassifier(vocab.get_size(), 300, 300, 2).to(device)
    sentence_classifier.load_state_dict(torch.load(checkpoint_path))

    total_count, right_count = 0, 0
    sentence_classifier.eval()
    valid_iterators = [get_eval_sample_iterator(path, 50) for path in paths]
    for i in range(len(valid_iterators)):
        for texts in valid_iterators[i]:
            labels = [i] * len(texts)
            total_count += len(labels)
            labels = torch.tensor(labels).long().to(device)
            text_ids, lengths = vocab.encode_sequence_batch(texts)
            text_ids = torch.tensor(text_ids).long().to(device)
            if type == "cnn":
                logits, probs = sentence_classifier.forward(text_ids)
            else:
                logits, probs = sentence_classifier.forward(text_ids, lengths)
            _, pred_labels = torch.max(probs, dim=1)
            right_count += torch.sum((labels == pred_labels).float()).tolist()
    print("[eval] acc=%.5f" % (right_count / total_count))


def build_attribute_classifier(vocab, checkpoint_path):
    sentence_classifier = AttentiveRNNClassifier(vocab.get_size(), 300, 300, 2).to(device)
    sentence_classifier.load_state_dict(torch.load(checkpoint_path))
    return sentence_classifier
