import random
import torch


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def get_sample_iterator(path, batch_size, max_steps, shuffle=False, buffer_size=1000):
    step = 0
    while step < max_steps:
        buffer = []
        for line in open(path, "r", encoding="utf-8"):
            buffer.append(line.strip())
            if len(buffer) >= buffer_size:
                if shuffle:
                    random.shuffle(buffer)
                yield buffer[:batch_size], step
                buffer = buffer[batch_size:]
                step += 1
        while len(buffer) > 0:
            if len(buffer) > batch_size:
                yield buffer[:batch_size], step
                buffer = buffer[batch_size:]
            else:
                yield buffer[:len(buffer)], step
                buffer = []
            step += 1


def get_sample_pair_iterator(path, batch_size, max_steps, shuffle=False, buffer_size=1000):
    step = 0

    while step < max_steps:
        buffer = []
        for line in open(path, "r", encoding="utf-8"):
            [sent0, sent1, sent2] = line.strip().split("\t")
            buffer.append([sent0, sent1, sent2])
            if len(buffer) >= buffer_size:
                if shuffle:
                    random.shuffle(buffer)
                batch = buffer[:batch_size]
                batch_0 = [item[0] for item in batch]
                batch_1 = [item[1] for item in batch]
                batch_2 = [item[2] for item in batch]
                yield batch_0, batch_1, batch_2, step
                buffer = buffer[batch_size:]
                step += 1

        while len(buffer) > 0:
            if len(buffer) > batch_size:
                batch = buffer[:batch_size]
                batch_0 = [item[0] for item in batch]
                batch_1 = [item[1] for item in batch]
                batch_2 = [item[2] for item in batch]
                yield batch_0, batch_1, batch_2, step
                buffer = buffer[batch_size:]
            else:
                batch = buffer[:len(buffer)]
                batch_0 = [item[0] for item in batch]
                batch_1 = [item[1] for item in batch]
                batch_2 = [item[2] for item in batch]
                yield batch_0, batch_1, batch_2, step
                buffer = []
            step += 1


def get_eval_sample_iterator(path, batch_size, buffer_size=1000):

    buffer = []
    for line in open(path, "r", encoding="utf-8"):
        buffer.append(line.strip())
        if len(buffer) >= buffer_size:
            yield buffer[:batch_size]
            buffer = buffer[batch_size:]
    while len(buffer) > 0:
        if len(buffer) > batch_size:
            yield buffer[:batch_size]
            buffer = buffer[batch_size:]
        else:
            yield buffer[:len(buffer)]
            buffer = []


def sort_batch_by_length(text_batch, ids_batch, length_batch, label_batch):
    batch = []
    for text, ids, length, label in zip(text_batch, ids_batch, length_batch, label_batch):
        batch.append([text, ids, length, label])
    batch = sorted(batch, key=lambda a: a[2], reverse=True)
    text_batch = [item[0] for item in batch]
    ids_batch = [item[1] for item in batch]
    ids_batch = torch.tensor(ids_batch).long().to(device)
    length_batch = [item[2] for item in batch]
    label_batch = [item[3] for item in batch]
    label_batch = torch.tensor(label_batch).long().to(device)
    return text_batch, ids_batch, length_batch, label_batch


def get_eval_sample_pair_iterator(path, ref_path, batch_size):
    multi_reference = False
    if isinstance(ref_path, list):
        multi_reference = True

    if multi_reference:
        path_list = [path] + ref_path
    else:
        path_list = [path, ref_path]
    path_list = [open(path, "r", encoding="utf-8") for path in path_list]
    # path_tuple = tuple(path_list)
    source_batch = []
    reference_batch = []
    while True:
        try:
            source_text = path_list[0].readline().strip()
            if source_text == "":
                break
            reference_text = [path.readline() for path in path_list[1:]]
            source_batch.append(source_text)
            reference_batch.append(reference_text)
            if len(source_batch) == batch_size:
                yield source_batch, reference_batch
                source_batch = []
                reference_batch = []
        except EOFError:
            break
    # for text_tuple in zip(path_tuple):
    #     text_list = list(text_tuple)
    #     source_text = text_list[0]
    #     reference_text = text_list[1:]
    #     source_batch.append(source_text)
    #     reference_batch.append(reference_text)
    #     if len(source_batch) == batch_size:
    #         yield source_batch, reference_batch
    #         source_batch = []
    #         reference_batch = []
    if len(source_batch) > 0:
        yield source_batch, reference_batch
