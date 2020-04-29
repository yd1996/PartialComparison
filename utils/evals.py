import collections
import math
from typing import Counter, List, Tuple, Union
import torch

__all__ = [
    "sentence_bleu",
    "corpus_bleu",
    "evaluate_attribute_transfer_model"
]


def _get_ngrams(segment,
                max_order: int) -> Counter[Tuple[str, ...]]:
    r"""Extracts all n-grams up to a given maximum order from an
    input segment.

    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned
            by this methods.

    Returns:
        The Counter containing all n-grams upto :attr:`max_order`
        in segment with a count of how many times each n-gram occurred.
    """
    ngram_counts: collections.Counter = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _maybe_str_to_list(list_or_str: Union[str, List[str]]) -> List[str]:
    if isinstance(list_or_str, str):
        return list_or_str.split()
    return list_or_str


def _lowercase(str_list: List[str]) -> List[str]:
    return [str_.lower() for str_ in str_list]


def sentence_bleu(references,
                  hypothesis,
                  max_order: int = 4,
                  lowercase: bool = False,
                  smooth: bool = False,
                  use_bp: bool = True,
                  return_all: bool = False):
    r"""Calculates BLEU score of a hypothesis sentence.

    Args:
        references: A list of reference for the hypothesis.
            Each reference can be either a list of string tokens, or a string
            containing tokenized tokens separated with whitespaces.
            List can also be numpy array.
        hypothesis: A hypothesis sentence.
            Each hypothesis can be either a list of string tokens, or a
            string containing tokenized tokens separated with whitespaces.
            List can also be numpy array.
        lowercase (bool): If `True`, lowercase reference and hypothesis
            tokens.
        max_order (int): Maximum n-gram order to use when computing
            BLEU score.
        smooth (bool): Whether or not to apply `(Lin et al. 2004)` smoothing.
        use_bp (bool): Whether to apply brevity penalty.
        return_all (bool): If `True`, returns BLEU and all
            n-gram precisions.

    Returns:
        If :attr:`return_all` is `False` (default), returns a float32
        BLEU score.

        If :attr:`return_all` is `True`, returns a list of float32
        ``[BLEU] + n-gram precisions``, which is of length :attr:`max_order`
        +1.
    """
    return corpus_bleu([references],
                       [hypothesis],
                       max_order=max_order,
                       lowercase=lowercase,
                       smooth=smooth,
                       use_bp=use_bp,
                       return_all=return_all)


def corpus_bleu(list_of_references,
                hypotheses,
                max_order: int = 4,
                lowercase: bool = False,
                smooth: bool = False,
                use_bp: bool = True,
                return_all: bool = False):
    r"""Computes corpus-level BLEU score.

    Args:
        list_of_references: A list of lists of references for each hypothesis.
            Each reference can be either a list of string tokens, or a string
            containing tokenized tokens separated with whitespaces.
            List can also be numpy array.
        hypotheses: A list of hypothesis sentences.
            Each hypothesis can be either a list of string tokens, or a
            string containing tokenized tokens separated with whitespaces.
            List can also be numpy array.
        lowercase (bool): If `True`, lowercase reference and hypothesis
            tokens.
        max_order (int): Maximum n-gram order to use when computing
            BLEU score.
        smooth (bool): Whether or not to apply `(Lin et al. 2004)` smoothing.
        use_bp (bool): Whether to apply brevity penalty.
        return_all (bool): If `True`, returns BLEU and all
            n-gram precisions.

    Returns:
        If :attr:`return_all` is `False` (default), returns a ``float32``
        BLEU score.

        If :attr:`return_all` is `True`, returns a list of
        ``float32`` scores: ``[BLEU] + n-gram precisions``,
        which is of length :attr:`max_order` +1.
    """

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    hypothesis_length = 0

    for (references, hypothesis) in zip(list_of_references, hypotheses):
        reference_length += min(len(r) for r in references)
        hypothesis_length += len(hypothesis)

        merged_ref_ngram_counts: Counter[Tuple[str, ...]] = \
            collections.Counter()
        for reference in references:
            reference = _maybe_str_to_list(reference)
            if lowercase:
                reference = _lowercase(reference)
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)

        hypothesis = _maybe_str_to_list(hypothesis)
        if lowercase:
            hypothesis = _lowercase(hypothesis)
        hypothesis_ngram_counts = _get_ngrams(hypothesis, max_order)

        overlap = hypothesis_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(hypothesis) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0.0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    if use_bp:
        ratio = float(hypothesis_length) / reference_length
        if ratio > 1.0:
            bp = 1.
        else:
            if abs(ratio) < 1e-8:
                bp = 0.
            else:
                bp = math.exp(1 - 1. / ratio)
    else:
        bp = 1.

    bleu = geo_mean * bp

    if return_all:
        return [bleu * 100] + [p * 100 for p in precisions]
    else:
        return bleu * 100


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def evaluate_attribute_transfer_model(hypos, src_texts, tgt_texts, vocab, target_labels,
                                      language_model, attribute_classifier):

    text_ids, lengths = vocab.encode_sequence_batch(hypos)
    text_ids = torch.tensor(text_ids).long().to(device)

    logits, probs = attribute_classifier.forward(text_ids, lengths)
    _, pred_labels = torch.max(probs, dim=1)
    right_count = torch.sum((target_labels == pred_labels).float()).tolist()
    total_count = len(hypos)

    bleu_ref = sum([sentence_bleu(ref, hypo) for hypo, ref in zip(hypos, tgt_texts)]) / total_count
    bleu_self = sum([sentence_bleu([ref], hypo) for hypo, ref in zip(hypos, src_texts)]) / total_count

    perplexity = sum(language_model.calculate_perplexity(text_ids)) / total_count

    return total_count, right_count, bleu_self, bleu_ref, perplexity
