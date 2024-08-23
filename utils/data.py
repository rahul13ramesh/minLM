import re
import torch
import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TextTrainDataset(Dataset):
    def __init__(self, filename, context_size):
        json_name = filename.replace('.npy', '.json')
        self.train = True

        with open(json_name, 'r') as f:
            info = json.load(f)

        self.eos_token = info['eos_token']
        self.context_size = context_size
        self.doc_len = info['doc_len'] - self.context_size - 1
        self.vocab_len = info['vocab_len']

        self.doc = np.memmap(filename, dtype=np.uint16, mode='r')

    def __len__(self):
        return self.doc_len

    def __getitem__(self, idx):

        idx1 = idx
        idx2 = idx + self.context_size + 1
        return torch.from_numpy(self.doc[idx1:idx2].astype(np.int64))


def wikitext103_loader(cfg, train=True):
    """
    Get dataloader for wikitext-103
    """
    if train:
        fname = cfg.data.train_file
    else:
        fname = cfg.data.val_file

    ds = TextTrainDataset(fname, cfg.net.context_size)

    if not train:
        bs = int(cfg.data.bs * 1.4)
    else:
        bs = cfg.data.bs

    loader = DataLoader(ds, batch_size=bs, shuffle=train,
                          pin_memory=True, num_workers=cfg.data.nworkers)

    return loader
