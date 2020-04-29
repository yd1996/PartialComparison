import os
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
from tensorboardX import SummaryWriter
from modules.encoder import RNNEncoder
from modules.decoder import RNNDecoder
from modules.discrminator import MLPDiscriminator
from modules.comparator import ContentComparator, StyleComparator
from modules.language_model import build_language_model
from modules.loss import calcuate_maximum_likehood_loss
from utils.dataloader import get_sample_iterator, get_sample_pair_iterator, get_eval_sample_pair_iterator
from utils.evals import evaluate_attribute_transfer_model


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
embedding_dim = 300
style_dim = 100
hidden_dim = 300


def train(params, vocab, language_model, attribute_classifier, checkpoint_dir):

    # model
    vocab_size = vocab.get_size()
    encoder_c = RNNEncoder(vocab_size, embedding_dim, hidden_dim, rnn_type="GRU").to(device)
    encoder_s = nn.Embedding(2, style_dim).to(device)
    generator = RNNDecoder(vocab_size, embedding_dim, hidden_dim + style_dim,
                           rnn_type="GRU", max_length=params["max_decoding_length"]).to(device)
    params_e = list(encoder_c.parameters()) + list(encoder_s.parameters())
    optimizer_e = Adam(params=params_e, lr=1e-4)
    optimizer_g = Adam(params=generator.parameters(), lr=1e-4)

    mlp_discriminator = MLPDiscriminator([hidden_dim, 128, 32], 2).to(device)
    optimizer_mlp = Adam(params=mlp_discriminator.parameters(), lr=1e-4)

    lm_discriminators = [build_language_model(vocab, params["language_model_checkpoint"][0]),
                         build_language_model(vocab, params["language_model_checkpoint"][1])]

    style_comparator = StyleComparator(vocab_size, embedding_dim,
                                       params["max_decoding_length"], [3, 4, 5]).to(device)
    optimizer_style = RMSprop(params=style_comparator.parameters(), lr=1e-4)

    content_comparator = ContentComparator(vocab_size, embedding_dim,
                                           params["max_decoding_length"], [3, 4, 5]).to(device)
    optimizer_content = RMSprop(params=content_comparator.parameters(), lr=1e-4)

    loss_func_d = nn.CrossEntropyLoss()

    if os.path.exists(checkpoint_dir + "content_comparator.pretrain"):
        content_comparator.load_state_dict(torch.load(checkpoint_dir + "content_comparator.pretrain"))
        style_comparator.load_state_dict(torch.load(checkpoint_dir + "style_comparator.pretrain"))
    else:
        train_iterators_c = get_sample_pair_iterator(params["pair_path"], 64, 100000 * params["iter_d"],
                                                     shuffle=True, buffer_size=100000)
        train_iterators_d = [get_sample_iterator(path, 64, 100000 * params["iter_d"], shuffle=True)
                             for path in params["train_paths"]]
        for _ in range(10000):
            # train content comparator
            texts_src, texts_1, texts_0, step = next(train_iterators_c)
            texts_src_ids, texts_src_lengths = vocab.encode_sequence_batch(
                texts_src, max_length=params["max_decoding_length"])
            texts_src_ids = torch.tensor(texts_src_ids).long().to(device)
            texts_1_ids, texts_1_lengths = vocab.encode_sequence_batch(
                texts_1, max_length=params["max_decoding_length"])
            texts_1_ids = torch.tensor(texts_1_ids).long().to(device)
            texts_0_ids, texts_0_lengths = vocab.encode_sequence_batch(
                texts_0, max_length=params["max_decoding_length"])
            texts_0_ids = torch.tensor(texts_0_ids).long().to(device)

            content_comparator.train()
            logits_1 = content_comparator.forward(texts_src_ids, texts_1_ids)
            logits_0 = content_comparator.forward(texts_src_ids, texts_0_ids)
            batch_size = len(texts_src)
            labels_1 = torch.ones(batch_size).long().to(device)
            labels_0 = torch.zeros(batch_size).long().to(device)
            loss_dis_c = loss_func_d(logits_1, labels_1) + loss_func_d(logits_0, labels_0) * 0.8

            # prepare data for mlp and style comparator
            texts_pair = []
            for i in range(len(train_iterators_d)):
                texts_, step = next(train_iterators_d[i])
                texts_pair.append(texts_)

            # train style comparator
            style_comparator.train()

            texts_l = texts_pair[0] + texts_pair[1]
            texts_r = texts_pair[1] + texts_pair[0]
            texts_l_ids, _ = vocab.encode_sequence_batch(texts_l, max_length=params["max_decoding_length"])
            texts_l_ids = torch.tensor(texts_l_ids).long().to(device)
            texts_r_ids, _ = vocab.encode_sequence_batch(texts_r, max_length=params["max_decoding_length"])
            texts_r_ids = torch.tensor(texts_r_ids).long().to(device)
            logits_1 = style_comparator.forward(texts_l_ids, texts_r_ids)
            labels_1 = torch.ones(texts_l_ids.size(0)).long().to(device)
            loss_dis_s_1 = loss_func_d(logits_1, labels_1)

            texts_r = texts_pair[0][1:] + texts_pair[0][0:1] + texts_pair[1][1:] + texts_pair[1][0:1]
            texts_r_ids, _ = vocab.encode_sequence_batch(texts_r, max_length=params["max_decoding_length"])
            texts_r_ids = torch.tensor(texts_r_ids).long().to(device)
            logits_0 = style_comparator.forward(texts_l_ids, texts_r_ids)
            labels_0 = torch.zeros(texts_l_ids.size(0)).long().to(device)
            loss_dis_s_0 = loss_func_d(logits_0, labels_0)

            loss_dis_s = loss_dis_s_1 + loss_dis_s_0

            # optimize two comparators
            optimizer_content.zero_grad()
            loss_dis_c = loss_dis_c / 2
            loss_dis_c.backward()
            optimizer_content.step()

            optimizer_style.zero_grad()
            loss_dis_s = loss_dis_s / 2
            loss_dis_s.backward()
            optimizer_style.step()

            print("[Pretrain Comparator] step=%d loss_s=%.5f loss_c=%.5f"
                  % (step, loss_dis_s, loss_dis_c))

        torch.save(style_comparator.state_dict(), checkpoint_dir + "style_comparator.pretrain")
        torch.save(content_comparator.state_dict(), checkpoint_dir + "content_comparator.pretrain")

    # pretrain auto-encoder
    if os.path.exists(checkpoint_dir + "encoder_c.pretrain"):
        encoder_c.load_state_dict(torch.load(checkpoint_dir + "encoder_c.pretrain"))
        encoder_s.load_state_dict(torch.load(checkpoint_dir + "encoder_s.pretrain"))
        generator.load_state_dict(torch.load(checkpoint_dir + "generator.pretrain"))
    else:
        train_iterators_g = [get_sample_iterator(path, 128, 20000, shuffle=True) for path in params["train_paths"]]
        for _ in range(10000):

            encoder_c.train()
            encoder_s.train()
            generator.train()

            loss_rec = 0
            loss_rec_val = 0

            for i in range(len(train_iterators_g)):
                texts, step = next(train_iterators_g[i])
                labels = [i] * len(texts)
                text_ids, lengths = vocab.encode_sequence_batch(texts)
                text_ids = torch.tensor(text_ids).long().to(device)
                labels = torch.tensor(labels).long().to(device)

                _, z_c = encoder_c(text_ids, lengths)
                z_s_r = encoder_s(labels)
                # reconstruction
                z_r = torch.cat([z_c, z_s_r], dim=-1)
                logits_r, states_r = generator.train_greedy_decode(text_ids, initial_state=z_r)

                # loss_rec
                loss_mle, loss_mle_val = calcuate_maximum_likehood_loss(logits_r, text_ids)
                loss_rec = loss_rec + loss_mle
                loss_rec_val += loss_mle_val.tolist()

            loss = loss_rec / 2
            loss_rec_val /= 2

            optimizer_e.zero_grad()
            optimizer_g.zero_grad()
            loss.backward()
            encoder_c.clip_grad()
            generator.clip_grad()
            optimizer_e.step()
            optimizer_g.step()

            display_line = "[train G] step=%d loss_rec=%.5f" % (step, loss_rec_val)
            print(display_line)
        torch.save(encoder_c.state_dict(), checkpoint_dir + "encoder_c.pretrain")
        torch.save(encoder_s.state_dict(), checkpoint_dir + "encoder_s.pretrain")
        torch.save(generator.state_dict(), checkpoint_dir + "generator.pretrain")

    # train
    train_iterators_g = [get_sample_iterator(path, 32, 100000 * params["iter_g"], shuffle=True)
                         for path in params["train_paths"]]
    train_iterators_c = get_sample_pair_iterator(params["pair_path"], 32, 100000 * params["iter_d"],
                                                 shuffle=True, buffer_size=100000)
    train_iterators_d = [get_sample_iterator(path, 32, 100000 * params["iter_d"], shuffle=True)
                         for path in params["train_paths"]]
    summary_writer = SummaryWriter(log_dir=checkpoint_dir)
    step = 0
    tau = 0.001
    while step < 200000:

        # training D
        encoder_c.eval()
        encoder_s.eval()
        generator.eval()

        for _ in range(params["iter_d"]):

            # train content comparator
            texts_src, texts_1, texts_0, step = next(train_iterators_c)
            texts_src_ids, texts_src_lengths = vocab.encode_sequence_batch(
                texts_src, max_length=params["max_decoding_length"])
            texts_src_ids = torch.tensor(texts_src_ids).long().to(device)
            texts_1_ids, texts_1_lengths = vocab.encode_sequence_batch(
                texts_1, max_length=params["max_decoding_length"])
            texts_1_ids = torch.tensor(texts_1_ids).long().to(device)
            texts_0_ids, texts_0_lengths = vocab.encode_sequence_batch(
                texts_0, max_length=params["max_decoding_length"])
            texts_0_ids = torch.tensor(texts_0_ids).long().to(device)

            content_comparator.train()
            logits_1 = content_comparator.forward(texts_src_ids, texts_1_ids)
            logits_0 = content_comparator.forward(texts_src_ids, texts_0_ids)
            batch_size = len(texts_src)
            labels_1 = torch.ones(batch_size).long().to(device)
            labels_0 = torch.zeros(batch_size).long().to(device)
            loss_dis_c = loss_func_d(logits_1, labels_1) + loss_func_d(logits_0, labels_0) * 0.8

            # prepare data for mlp and style comparator
            texts, labels = [], []
            texts_pair = []
            for i in range(len(train_iterators_d)):
                texts_, step = next(train_iterators_d[i])
                texts.extend(texts_)
                texts_pair.append(texts_)
                labels_ = [i] * len(texts_)
                labels.extend(labels_)
            text_ids, lengths = vocab.encode_sequence_batch(texts)
            text_ids = torch.tensor(text_ids).long().to(device)
            labels = torch.tensor(labels).long().to(device)

            # train style comparator
            style_comparator.train()

            texts_l = texts_pair[0] + texts_pair[1]
            texts_r = texts_pair[1] + texts_pair[0]
            texts_l_ids, _ = vocab.encode_sequence_batch(texts_l, max_length=params["max_decoding_length"])
            texts_l_ids = torch.tensor(texts_l_ids).long().to(device)
            texts_r_ids, _ = vocab.encode_sequence_batch(texts_r, max_length=params["max_decoding_length"])
            texts_r_ids = torch.tensor(texts_r_ids).long().to(device)
            logits_1 = style_comparator.forward(texts_l_ids, texts_r_ids)
            labels_1 = torch.ones(texts_l_ids.size(0)).long().to(device)
            loss_dis_s_1 = loss_func_d(logits_1, labels_1)

            texts_r = texts_pair[0][1:] + texts_pair[0][0:1] + texts_pair[1][1:] + texts_pair[1][0:1]
            texts_r_ids, _ = vocab.encode_sequence_batch(texts_r, max_length=params["max_decoding_length"])
            texts_r_ids = torch.tensor(texts_r_ids).long().to(device)
            logits_0 = style_comparator.forward(texts_l_ids, texts_r_ids)
            labels_0 = torch.zeros(texts_l_ids.size(0)).long().to(device)
            loss_dis_s_0 = loss_func_d(logits_0, labels_0)

            loss_dis_s = loss_dis_s_1 + loss_dis_s_0 * 0.8

            # train mlp
            mlp_discriminator.train()
            _, z_c = encoder_c(text_ids, lengths)
            z_c = z_c.detach()
            logits_z, _ = mlp_discriminator(z_c)
            loss_dis_z = loss_func_d(logits_z, labels)
            optimizer_mlp.zero_grad()
            loss_dis_z.backward()
            optimizer_mlp.step()

            # generate negative samples
            z_s = encoder_s(1 - labels)
            z = torch.cat([z_c, z_s], dim=-1)
            outputs = generator.beam_search(3, initial_state=z)
            output_texts = vocab.decode_sequence_batch(outputs)
            output_ids, _ = vocab.encode_sequence_batch(output_texts, max_length=params["max_decoding_length"])
            output_ids = torch.tensor(output_ids).long().to(device)
            logits_c = content_comparator.forward(text_ids, output_ids)
            loss_dis_c = loss_dis_c + loss_func_d(logits_c, labels_0) * 0.2
            logits_s = style_comparator.forward(text_ids, output_ids)
            loss_dis_s = loss_dis_s + loss_func_d(logits_s, labels_0) * 0.2

            # optimize two comparators
            optimizer_content.zero_grad()
            loss_dis_c = loss_dis_c / 2
            loss_dis_c.backward()
            optimizer_content.step()

            optimizer_style.zero_grad()
            loss_dis_s = loss_dis_s / 2
            loss_dis_s.backward()
            optimizer_style.step()

            print("[Training D] step=%d loss_dis_c=%.5f loss_dis_s=%.5f loss_dis_z=%.5f"
                  % (step, loss_dis_c, loss_dis_s, loss_dis_z))
            summary_writer.add_scalar("Loss/Loss_dis_c", loss_dis_c, global_step=step)
            summary_writer.add_scalar("Loss/Loss_dis_s", loss_dis_s, global_step=step)
            summary_writer.add_scalar("Loss/Loss_dis_z", loss_dis_z, global_step=step)

        # training G
        for _ in range(params["iter_g"]):

            encoder_c.train()
            encoder_s.train()
            generator.train()

            loss_rec = 0
            loss_rec_val = 0
            loss_adv_z = 0
            loss_adv_c = 0
            loss_adv_s = 0
            loss_adv_lm = 0

            for i in range(len(train_iterators_g)):
                texts, step = next(train_iterators_g[i])
                labels = [i] * len(texts)
                text_ids, lengths = vocab.encode_sequence_batch(texts)
                text_ids = torch.tensor(text_ids).long().to(device)
                labels = torch.tensor(labels).long().to(device)

                _, z_c = encoder_c(text_ids, lengths)
                z_s_r = encoder_s(labels)
                # reconstruction
                z_r = torch.cat([z_c, z_s_r], dim=-1)
                logits_r, states_r = generator.train_greedy_decode(text_ids, initial_state=z_r)

                # attribute transfer
                z_s_t = encoder_s(1 - labels)
                z_t = torch.cat([z_c, z_s_t], dim=-1).detach()
                logits_t, outputs_t, states_t, lengths_t = generator.train_gumbel_softmax_decode(tau, z_t)

                # loss_rec
                loss_mle, loss_mle_val = calcuate_maximum_likehood_loss(logits_r, text_ids)
                loss_rec = loss_rec + loss_mle
                loss_rec_val += loss_mle_val.tolist()

                # loss_adv_z
                mlp_discriminator.train()
                logits_z, _ = mlp_discriminator(z_c)
                loss_adv_z = loss_adv_z + loss_func_d(logits_z, labels)

                # loss_adv_lm
                lm_discriminators[1 - i].train()
                rewards = lm_discriminators[1 - i].calculate_reward_signal(outputs_t.detach())
                entropy = - outputs_t * rewards.detach()
                entropy = torch.sum(entropy, dim=-1)
                mask = torch.zeros_like(entropy).byte()
                for j in range(len(lengths)):
                    for k in range(lengths[j]):
                        mask[j, k] = 1
                entropy = entropy[mask]
                entropy = torch.mean(entropy)
                loss_adv_lm = loss_adv_lm + entropy

                labels_real = torch.ones_like(labels)

                # loss_adv_s
                style_comparator.train()
                logits = style_comparator.forward(text_ids, outputs_t)
                loss_adv_s = loss_adv_s + loss_func_d(logits, labels_real)

                # loss_adv_c
                content_comparator.train()
                logits = content_comparator.forward(text_ids, outputs_t)
                loss_adv_c = loss_adv_c + loss_func_d(logits, labels_real)

            loss_rec = loss_rec / 2
            loss_rec_val /= 2
            loss_adv_z = loss_adv_z / 2
            loss_adv_lm = loss_adv_lm / 2
            loss_adv_c = loss_adv_c / 2
            loss_adv_s = loss_adv_s / 2
            loss = loss_rec - params["lambda_adv_z"] * loss_adv_z \
                   + params["lambda_adv_lm"] * loss_adv_lm \
                   + params["lambda_adv_c"] * loss_adv_c \
                   + params["lambda_adv_s"] * loss_adv_s

            optimizer_e.zero_grad()
            optimizer_g.zero_grad()
            loss.backward()
            encoder_c.clip_grad()
            generator.clip_grad()
            optimizer_e.step()
            optimizer_g.step()

            print("[Training G] step=%d loss_rec=%.5f loss_adv_z=%.5f loss_adv_c=%.5f "
                  "loss_adv_s=%.5f loss_adv_lm=%.5f"
                  % (step, loss_rec_val, loss_adv_z, loss_adv_c, loss_adv_s, loss_adv_lm))
            summary_writer.add_scalar("Loss/Loss_rec", loss_rec_val, global_step=step)
            summary_writer.add_scalar("Loss/Loss_adv_z", loss_adv_z, global_step=step)
            summary_writer.add_scalar("Loss/Loss_adv_c", loss_adv_c, global_step=step)
            summary_writer.add_scalar("Loss/Loss_adv_s", loss_adv_s, global_step=step)
            summary_writer.add_scalar("Loss/Loss_adv_lm", loss_adv_lm, global_step=step)

            if step % 50 == 0:
                encoder_c.eval()
                encoder_s.eval()
                generator.eval()
                total_count = 0
                right_count = 0
                bleu_self, bleu_ref = 0, 0
                perplexity = 0
                valid_iterators = [get_eval_sample_pair_iterator(path_pair["src"], path_pair["ref"], 20)
                                   for path_pair in params["valid_path_pairs"]]
                for i in range(len(valid_iterators)):
                    for texts, refs in valid_iterators[i]:
                        text_ids, lengths = vocab.encode_sequence_batch(texts)
                        text_ids = torch.tensor(text_ids).long().to(device)
                        target_labels = torch.tensor([1 - i] * len(lengths)).long().to(device)
                        state_s = encoder_s(target_labels)
                        _, state_c = encoder_c(text_ids, lengths)
                        state = torch.cat([state_c, state_s], dim=-1)
                        outputs = generator.greedy_search(initial_state=state)
                        hypos = vocab.decode_sequence_batch(outputs)
                        total_count_, right_count_, bleu_self_, bleu_ref_, perplexity_ = \
                            evaluate_attribute_transfer_model(
                                hypos, texts, refs, vocab, target_labels, language_model, attribute_classifier)
                        total_count += total_count_
                        right_count += right_count_
                        bleu_self += bleu_self_ * total_count_
                        bleu_ref += bleu_ref_ * total_count_
                        perplexity += perplexity_ * total_count_
                accuracy = right_count / total_count
                bleu_self = bleu_self / total_count
                bleu_ref = bleu_ref / total_count
                perplexity = perplexity / total_count
                print("[eval] step=%d acc=%.4f self-bleu=%.4f ref-bleu=%.4f ppl=%.4f"
                      % (step, accuracy, bleu_self, bleu_ref, perplexity))
                summary_writer.add_scalar("Eval/Accuracy", accuracy, global_step=step)
                summary_writer.add_scalar("Eval/self-BLEU", bleu_self, global_step=step)
                summary_writer.add_scalar("Eval/ref-BLEU", bleu_ref, global_step=step)
                summary_writer.add_scalar("Eval/Perplexity", perplexity, global_step=step)

            if step > 0 and step % 5000 == 0:
                torch.save(encoder_s.state_dict(), checkpoint_dir + "encoder_s.%d" % step)
                torch.save(encoder_c.state_dict(), checkpoint_dir + "encoder_c.%d" % step)
                torch.save(generator.state_dict(), checkpoint_dir + "generator.%d" % step)
                torch.save(mlp_discriminator.state_dict(), checkpoint_dir + "mlp_d.%d" % step)
                torch.save(style_comparator.state_dict(), checkpoint_dir + "comparator_s.%d" % step)
                torch.save(content_comparator.state_dict(), checkpoint_dir + "comparator_c.%d" % step)

        # if step % 10000 == 0 and tau > 1e-4:
        #     tau *= 0.5

    summary_writer.close()


def evaluate(params, vocab, language_model, attribute_classifier, checkpoint_dir):

    vocab_size = vocab.get_size()
    encoder_c = RNNEncoder(vocab_size, embedding_dim, hidden_dim, rnn_type="GRU").to(device)
    encoder_s = nn.Embedding(2, style_dim).to(device)
    generator = RNNDecoder(vocab_size, embedding_dim, hidden_dim + style_dim,
                           rnn_type="GRU", max_length=params["max_decoding_length"]).to(device)

    paths = os.listdir(checkpoint_dir)
    index_list = []
    for path in paths:
        if path.startswith("encoder_s") and not path.endswith("pretrain"):
            index_list.append(int(path.strip().split(".")[1]))
    index_list.sort()

    for index in index_list:
        encoder_s.load_state_dict(torch.load(checkpoint_dir + "encoder_s.%d" % index))
        encoder_c.load_state_dict(torch.load(checkpoint_dir + "encoder_c.%d" % index))
        generator.load_state_dict(torch.load(checkpoint_dir + "generator.%d" % index))

        encoder_c.eval()
        encoder_s.eval()
        generator.eval()

        total_count = 0
        right_count = 0
        bleu_self, bleu_ref = 0, 0
        perplexity = 0
        valid_iterators = [get_eval_sample_pair_iterator(path_pair["src"], path_pair["ref"],
                                                         params["max_decoding_length"])
                           for path_pair in params["valid_path_pairs"]]
        writer = open(checkpoint_dir + "test.%d" % index, "w", encoding="utf-8")
        for i in range(len(valid_iterators)):
            for texts, refs in valid_iterators[i]:
                text_ids, lengths = vocab.encode_sequence_batch(texts)
                text_ids = torch.tensor(text_ids).long().to(device)
                target_labels = torch.tensor([1 - i] * len(lengths)).long().to(device)
                state_s = encoder_s(target_labels)
                _, state_c = encoder_c(text_ids, lengths)
                state = torch.cat([state_c, state_s], dim=-1)
                outputs = generator.beam_search(3, initial_state=state)
                hypos = vocab.decode_sequence_batch(outputs)
                for text, hypo in zip(texts, hypos):
                    writer.write(text + " --> " + hypo + "\n")
                    print(index, text, "-->", hypo)
                total_count_, right_count_, bleu_self_, bleu_ref_, perplexity_ = \
                    evaluate_attribute_transfer_model(
                        hypos, texts, refs, vocab, target_labels, language_model, attribute_classifier)
                total_count += total_count_
                right_count += right_count_
                bleu_self += bleu_self_ * total_count_
                bleu_ref += bleu_ref_ * total_count_
                perplexity += perplexity_ * total_count_
        accuracy = right_count / total_count
        bleu_self = bleu_self / total_count
        bleu_ref = bleu_ref / total_count
        perplexity = perplexity / total_count
        writer.write("accuracy: ")
        writer.write(str(accuracy))
        writer.write(" self-bleu:")
        writer.write(str(bleu_self))
        writer.write(" ref-bleu:")
        writer.write(str(bleu_ref))
        writer.write(" ppl:")
        writer.write(str(perplexity))
        writer.write("\n")
        writer.close()
