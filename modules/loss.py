import torch
from torch.nn.functional import cross_entropy
from utils.vocab import SpecialTokens


def calcuate_maximum_likehood_loss(logits, labels, pad_id=SpecialTokens.pad_index):

    batch_size, max_length, vocab_size = logits.size(0), logits.size(1), logits.size(2)
    flatten_logits = logits.view(batch_size * max_length, vocab_size)
    flatten_labels = labels.view(batch_size * max_length)
    losses = cross_entropy(flatten_logits, flatten_labels, ignore_index=pad_id, reduce=False)
    mask = flatten_labels != pad_id
    loss_sum = torch.sum(losses[mask])
    loss_size = torch.sum(mask.float()).tolist()
    loss_mean = loss_sum / loss_size
    loss_sum = loss_sum / batch_size

    return loss_sum, loss_mean


def calculate_perplexity(logits, labels, pad_id=SpecialTokens.pad_index):

    batch_size, max_length, vocab_size = logits.size(0), logits.size(1), logits.size(2)
    flatten_logits = logits.view(batch_size * max_length, vocab_size)
    flatten_labels = labels.view(batch_size * max_length)
    losses = cross_entropy(flatten_logits, flatten_labels, ignore_index=pad_id, reduce=False)
    losses = losses.view(batch_size, max_length)
    losses_sum = torch.sum(losses, dim=1)
    losses_length = torch.sum((labels != pad_id).float(), dim=1)
    perplexity = torch.exp(losses_sum / losses_length)

    return perplexity


def calculate_gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(
        1 + (recog_logvar - prior_logvar)
        - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
        - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), dim=-1)
    return kld