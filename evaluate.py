import argparse
import json
import torch
from utils.vocab import Vocab
from modules.language_model import build_language_model
from modules.classifier import build_attribute_classifier
from utils.dataloader import get_eval_sample_pair_iterator
from utils.evals import evaluate_attribute_transfer_model

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset", type=str, default="yelp")
argparser.add_argument("--path_0", type=str, default="result.0")
argparser.add_argument("--path_1", type=str, default="result.1")
args = argparser.parse_args()

print("dataset:", args.dataset)
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

if args.dataset == "yelp":
    params = "data/config/cross_alignment/yelp.json"
else:
    params = "data/config/cross_alignment/amazon.json"
params = json.load(open(params, "r", encoding="utf-8"))
vocab = Vocab(params["vocab_path"], params["max_vocab_size"], params["min_token_freq"])
print("vocab_size=%d" % vocab.get_size())
language_model = build_language_model(vocab, params["eval_language_model_path"])
print("language model has been loaded")
attribute_classifier = build_attribute_classifier(vocab, params["eval_attribute_classifier_path"])
print("attribute classifier has been loaded")

path_pair = params["valid_path_pairs"]
eval_iterators = [get_eval_sample_pair_iterator(args.path_0, [path_pair[0]["src"]] + path_pair[0]["ref"],
                                                params["max_decoding_length"]),
                  get_eval_sample_pair_iterator(args.path_1, [path_pair[1]["src"]] + path_pair[1]["ref"],
                                                params["max_decoding_length"])]
total_count = 0
right_count = 0
bleu_self, bleu_ref = 0, 0
perplexity = 0
for label in range(len(eval_iterators)):
    for texts, refs in eval_iterators[label]:
        _, lengths = vocab.encode_sequence_batch(texts)
        target_labels = torch.tensor([1 - label] * len(lengths)).long().to(device)
        src_texts = [item[0].strip() for item in refs]
        ref_texts = [item[1:] for item in refs]
        total_count_, right_count_, bleu_self_, bleu_ref_, perplexity_ = \
            evaluate_attribute_transfer_model(
                texts, src_texts, ref_texts, vocab, target_labels, language_model, attribute_classifier)
        total_count += total_count_
        right_count += right_count_
        bleu_self += bleu_self_ * total_count_
        bleu_ref += bleu_ref_ * total_count_
        perplexity += perplexity_ * total_count_

accuracy = right_count / total_count
bleu_self = bleu_self / total_count
bleu_ref = bleu_ref / total_count
perplexity = perplexity / total_count
print("acc=%.4f self-bleu=%.4f ref-bleu=%.4f ppl=%.4f" % (accuracy, bleu_self, bleu_ref, perplexity))
