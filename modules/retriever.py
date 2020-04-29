import os
import json
import time
import random
import numpy as np
from nltk import pos_tag
from utils.evals import sentence_bleu


def pretrain_word2vec(paths, save_dir):
    count = 0
    lines = []
    for path in paths:
        file = open(path, "r", encoding="utf-8")
        for line in file:
            lines.append(line)
            count += 1
            print(path, count)
        file.close()
    writer = open(save_dir + "data.merge", "w", encoding="utf-8")
    random.shuffle(lines)
    for line in lines:
        writer.write(line)
    writer.close()
    from gensim.models.word2vec import Word2Vec, LineSentence
    start = time.time()
    word2vec = Word2Vec(sentences=LineSentence(save_dir + "data.merge"), iter=15)
    end = time.time()
    print("Training time:", end - start)
    embedding_dict = dict()
    for token in word2vec.wv.index2word:
        embedding_dict[token] = word2vec[token].tolist()
        print(token)
    open(save_dir + "embedding.dict", "w", encoding="utf-8").write(json.dumps(embedding_dict))
    os.remove(save_dir + "data.merge")


def build_stop_words_table(vocab_path, output_path):
    from nltk import pos_tag
    writer = open(output_path, "w", encoding="utf-8")
    for line in open(vocab_path, "r", encoding="utf-8"):
        [token, freq] = line.strip().split("\t")
        if int(freq) <= 1000:
            break
        _, tag = pos_tag([token])[0]
        if tag.startswith("JJ") or tag.startswith("RB"):
            continue
        if not tag.startswith("NN"):
            writer.write(token + "\n")
    writer.close()


# build_stop_words_table("../data/corpus/yelp/vocab", "../data/corpus/yelp/stop_words")


class Retriever(object):

    def __init__(self, save_dir, embedding_path):

        self.save_dir = save_dir
        self.tokens = set()
        self.dictionary = dict()
        self.embedding = json.load(open(embedding_path, "r", encoding="utf-8"))
        self.embedding_tokens = set(list(self.embedding.keys()))

    def embed(self, text):
        embedding = None
        size = 0
        for token in text.split():
            if token in self.embedding_tokens:
                if embedding is None:
                    embedding = np.array(self.embedding[token])
                else:
                    embedding += np.array(self.embedding[token])
                size += 1
        if size == 0:
            return None
        else:
            embedding = embedding / size
            return embedding

    def build_index(self, data_paths, vocab_path, stop_word_path, save_path):

        stop_words = [line.strip() for line in open(stop_word_path, "r", encoding="utf-8")]
        stop_words = set(stop_words)
        for line in open(vocab_path, "r", encoding="utf-8"):
            [token, _] = line.strip().split("\t")
            if token in stop_words:
                continue
            self.tokens.add(token)
            self.dictionary[token] = set()

        count = 0
        for path in data_paths:
            for line in open(path, "r", encoding="utf-8"):
                text = line.strip()
                for token in text.split():
                    if token in self.tokens:
                        self.dictionary[token].add(text)
                count += 1
                print(count)

        dictionary = dict()
        for key, value in self.dictionary.items():
            dictionary[key] = list(value)
        json.dump(dictionary, open(save_path, "w", encoding="utf-8"))

    def load_index(self, save_path):

        dictionary = json.load(open(save_path, "r", encoding="utf-8"))
        for key, value in dictionary.items():
            self.dictionary[key] = set(value)
            self.tokens.add(key)

    def retrieve_text_with_token_overlap(self, query):

        nn_set = set()
        for token, tag in pos_tag(query.split()):
            if tag.startswith("NN"):
                nn_set.add(token)
        candidates = []
        if len(nn_set) > 0:
            candidate_set = []
            for token in nn_set:
                if token in self.tokens:
                    candidate_set.append(self.dictionary[token])
            if len(candidate_set) > 0:
                intersect_set = candidate_set[0]
                for i in range(1, len(candidate_set)):
                    intersect_set = intersect_set & candidate_set[i]
                if len(intersect_set) > 0:
                    candidates = list(intersect_set)
                else:
                    for i in range(len(candidate_set)):
                        candidates.extend(list(candidate_set[i]))
                    candidates = list(set(candidates))
        if len(candidates) <= 100:
            for token in query.split():
                if token in self.tokens:
                    candidates = candidates + list(self.dictionary[token])
            candidates = list(set(candidates))
        if len(candidates) < 100:
            keys = list(self.tokens)
            key = keys[random.randint(0, len(keys) - 1)]
            candidates = candidates + list(self.dictionary[key])

        candidate_score_pairs = []
        query_emb = self.embed(query)
        query_emb_norm = np.linalg.norm(query_emb)
        if query_emb is not None:
            count = 0
            for candidate in candidates:
                if candidate == query:
                    continue
                intersect_len = len(set(candidate.split()) & set(query.split()))
                condition = float(intersect_len) / float(len(query.split())) >= 0.5
                if condition or count < 100:
                    candidate_emb = self.embed(candidate)
                    if candidate_emb is not None:
                        score = np.dot(query_emb, candidate_emb) \
                                / (query_emb_norm * np.linalg.norm(candidate_emb))
                        score = score.tolist()
                        candidate_score_pairs.append([candidate, score])
                        count += 1

        best_candidate, best_score = " ", -1.0
        for [candidate, score] in candidate_score_pairs:
            if score > best_score and candidate != query:
                best_candidate, best_score = candidate, score
        return best_candidate, best_score, candidate_score_pairs

    def retrieve_text_with_BLEU(self, query):

        nn_set = set()
        for token, tag in pos_tag(query.split()):
            if tag.startswith("NN"):
                nn_set.add(token)
        candidates = []
        if len(nn_set) > 0:
            candidate_set = []
            for token in nn_set:
                if token in self.tokens:
                    candidate_set.append(self.dictionary[token])
            if len(candidate_set) > 0:
                intersect_set = candidate_set[0]
                for i in range(1, len(candidate_set)):
                    intersect_set = intersect_set & candidate_set[i]
                if len(intersect_set) > 0:
                    candidates = list(intersect_set)
                else:
                    for i in range(len(candidate_set)):
                        candidates.extend(list(candidate_set[i]))
                    candidates = list(set(candidates))
        if len(candidates) <= 100:
            for token in query.split():
                if token in self.tokens:
                    candidates = candidates + list(self.dictionary[token])
            candidates = list(set(candidates))
        if len(candidates) < 100:
            keys = list(self.tokens)
            key = keys[random.randint(0, len(keys) - 1)]
            candidates = candidates + list(self.dictionary[key])

        positive_samples = []
        negative_samples = []
        for candidate in candidates:
            if candidate == query:
                continue
            intersect_len = len(set(candidate.split()) & set(query.split()))
            condition = float(intersect_len) / float(len(query.split())) >= 0.5
            if condition:
                bleu = sentence_bleu([query], candidate) + sentence_bleu([candidate], query)
                bleu /= 2
                if bleu > 30:
                    positive_samples.append(candidate)
                else:
                    negative_samples.append(candidate)
        return positive_samples, negative_samples

