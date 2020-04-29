import os
import argparse
import json
from utils.vocab import Vocab
import modules.language_model as ppl_metric
import modules.classifier as acc_metric


argparser = argparse.ArgumentParser()
argparser.add_argument("--metric", type=str, default="ppl")
argparser.add_argument("--mode", type=str, default="train")
argparser.add_argument("--params", type=str, default="yelp.json")
argparser.add_argument("--checkpoint_dir", type=str, default="data/checkpoint/yelp/language_model_pos/")
argparser.add_argument("--checkpoint_path", type=str, default="")
argparser.add_argument("--max_length", type=int, default=20)
argparser.add_argument("--classifier", type=str, default="cnn")
args = argparser.parse_args()

if args.metric == "ppl":
    params_dir = "data/config/language_model/"
elif args.metric == "acc":
    params_dir = "data/config/sentence_classifier/"
else:
    raise ValueError("invalid metric!")
print(args.metric, args.mode)

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

params_path = params_dir + args.params
params = json.load(open(params_path, "r", encoding="utf-8"))
vocab = Vocab(params["vocab_path"], params["max_vocab_size"], params["min_token_freq"])
print("vocab_size=%d" % vocab.get_size())

if args.metric == "ppl":
    if args.mode == "train":
        ppl_metric.train_language_model(params["train_paths"], params["valid_paths"], vocab, args.checkpoint_dir)
    elif args.mode == "eval":
        ppl_metric.evaluate(params["test_paths"], vocab, args.checkpoint_path)
    else:
        raise ValueError("invalid mode!")
elif args.metric == "acc":
    if args.mode == "train":
        acc_metric.train_sentence_classifier(params["train_paths"], params["valid_paths"],
                                             vocab, args.max_length, args.checkpoint_dir, type=args.classifier)
    elif args.mode == "eval":
        acc_metric.evaluate(params["test_paths"], vocab, args.max_length, args.checkpoint_path, type=args.classifier)
    else:
        raise ValueError("invalid mode!")
