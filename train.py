import os
import argparse
import json
from utils.vocab import Vocab
from modules.language_model import build_language_model
from modules.classifier import build_attribute_classifier
import models.cross_alignment as cross_alignment
import models.style_transformer as style_transformer
import models.multi_decoders as multi_decoders
import models.controlled_generator as control_gen
import models.partial_comparison as partial_comparison
import models.SentiGAN as SentiGAN

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="cross_alignment")
argparser.add_argument("--run_mode", type=str, default="train")
argparser.add_argument("--params", type=str, default="cross_alignment/yelp.json")
argparser.add_argument("--checkpoint_dir", type=str, default="data/checkpoint/yelp/cross_alignment/")
args = argparser.parse_args()

print("model=%s run_mode=%s" % (args.model, args.run_mode))

params = json.load(open("data/config/" + args.params, "r", encoding="utf-8"))
vocab = Vocab(params["vocab_path"], params["max_vocab_size"], params["min_token_freq"])
print("vocab_size=%d" % vocab.get_size())

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

language_model = build_language_model(vocab, params["eval_language_model_path"])
print("language model has been loaded")
attribute_classifier = build_attribute_classifier(vocab, params["eval_attribute_classifier_path"])
print("attribute classifier has been loaded")

if args.model == "partial_comparison":
    if args.run_mode == "train":
        partial_comparison.train(params, vocab, language_model, attribute_classifier, args.checkpoint_dir)
    else:
        partial_comparison.evaluate(params, vocab, language_model, attribute_classifier, args.checkpoint_dir)

