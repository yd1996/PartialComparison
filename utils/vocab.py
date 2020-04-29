class SpecialTokens:

    pad_index = 0
    sos_index = 1
    eos_index = 2
    unk_index = 3
    pad_token = "<pad>"
    sos_token = "<sos>"
    eos_token = "<eos>"
    unk_token = "<unk>"


def build_vocab(in_paths, out_path):

    token2freq = dict()
    tokens_set = set()
    for path in in_paths:
        for line in open(path):
            tokens = line.split()
            for token in tokens:
                if token in tokens_set:
                    token2freq[token] += 1
                else:
                    token2freq[token] = 1
                    tokens_set.add(token)
                    print(len(tokens_set))
    token_freq = [[token, freq] for token, freq in token2freq.items()]
    token_freq = sorted(token_freq, key=lambda a: a[1], reverse=True)
    writer = open(out_path, "w", encoding="utf-8")
    for [token, freq] in token_freq:
        writer.write(token)
        writer.write("\t")
        writer.write(str(freq))
        writer.write("\n")
    writer.close()


class Vocab:

    def __init__(self, path, max_size, min_freq):

        self.token2index = {
            SpecialTokens.pad_token: 0, SpecialTokens.sos_token: 1,
            SpecialTokens.eos_token: 2, SpecialTokens.unk_token: 3
        }
        self.index2token = [
            SpecialTokens.pad_token, SpecialTokens.sos_token,
            SpecialTokens.eos_token, SpecialTokens.unk_token
        ]
        self.tokens = set()
        for line in open(path, "r", encoding="utf-8"):
            [token, freq] = line.strip().split("\t")
            freq = int(freq)
            if freq < min_freq or len(self.index2token) >= max_size + 4:
                break
            self.token2index[token] = len(self.index2token)
            self.index2token.append(token)
            self.tokens.add(token)
        self.size = len(self.index2token)

    def get_size(self):
        return self.size

    def encode_sequence(self, sequence, max_length=20):

        if isinstance(sequence, str):
            tokens = sequence.split()
        else:
            tokens = sequence

        ids = []
        for token in tokens:
            if len(ids) == max_length:
                break
            if token in self.tokens:
                ids.append(self.token2index[token])
            else:
                ids.append(SpecialTokens.unk_index)
        ids.append(SpecialTokens.eos_index)
        length = len(ids)

        return ids, length

    def encode_sequence_batch(self, batch, max_length=20):

        sequences, lengths = [], []
        for tokens in batch:
            ids, length = self.encode_sequence(tokens, max_length=max_length)
            sequences.append(ids)
            lengths.append(length)
        max_seq_len = max(lengths)
        for i in range(len(sequences)):
            while len(sequences[i]) < max_seq_len:
                sequences[i].append(SpecialTokens.pad_index)

        return sequences, lengths

    def decode_sequence(self, ids):
        tokens = []
        for id in ids:
            if id == SpecialTokens.eos_index:
                break
            tokens.append(self.index2token[id])
        sequence = " ".join(tokens)
        return sequence

    def decode_sequence_batch(self, batch):
        sequence_batch = [self.decode_sequence(ids) for ids in batch]
        return sequence_batch
