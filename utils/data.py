import re
import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TextTrainDataset(Dataset):
    def __init__(self, filename, context_size):
        json_name = filename.replace('.npy', '.json')
        with open(json_name, 'r') as f:
            info = json.load(f)

        self.doc_len = info['doc_len']
        self.eos_token = info['eos_token']
        self.context_size = context_size
        self.vocab_len = info['vocab_len']

        self.doc = np.memmap(filename, dtype=np.uint16, mode='r', shape=(self.doc_len,))

    def __len__(self):
        return self.doc_len

    def __getitem__(self, idx):
        idx1 = idx - self.context_size
        idx2 = idx + self.context_size + 1
        return self.doc[idx1:idx2]


def wikitext103_loader(cfg, train=True):
    """
    Get dataloader for wikitext-103
    """
    if train:
        fname = cfg.data.train_file
    else:
        fname = cfg.data.val_file

    ds = TextTrainDataset(fname, cfg.net.context_size)

    loader = DataLoader(ds, batch_size=cfg.data.bs, shuffle=train,
                          pin_memory=True, num_workers=cfg.data.nworkers)

    return loader
