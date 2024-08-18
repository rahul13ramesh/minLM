import re
import os
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def wikitext103_loader(cfg, train=True):

    split = "train" if train else "test"

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split=split)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    if train:
        if cfg.data.subsample < 1.0:
            nsamples = int(len(ds) * cfg.data.subsample)
            ds = ds.select(range(nsamples))

        # Remove empty examples
        ds = ds.filter(lambda example: (len(example['text']) > 0) and 
                                        (not re.match(r'( =)+.*?(= )+\n', example['text'])))


    def tokenize(txt):
        tokenized = tokenizer(txt['text'], truncation=True,
                              padding='max_length', 
                              max_length=cfg.net.context_size,
                              return_tensors='pt')
        return tokenized

    ds = ds.map(tokenize, batched=True)

    def len_filter(ex):
        ln = np.where(np.array(ex['input_ids']) == tokenizer.pad_token_id)[0]
        ln = ln[0] if len(ln) > 0 else 21

        return ln > 20

    # ds = ds.filter(lambda example: len(example['input_ids']) > 20)
    ds = ds.filter(len_filter)
    ds.set_format(type='torch', columns=['input_ids'])

    loader = DataLoader(ds, batch_size=cfg.data.bs, shuffle=train,
                          pin_memory=True, num_workers=4)

    return loader

