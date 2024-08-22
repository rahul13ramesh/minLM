import tiktoken
import functools
import json
import tqdm
import os
import numpy as np
from datasets import load_dataset


def encode_dataset(dataset, process, enc, fname,
                    eos='\n', subsample=1.0, batches=1,):
    # Subsample dataset
    eos_token = enc.encode(eos)[0]

    if subsample < 1.0:
        nsamples = int(len(dataset) * subsample)
        dataset = dataset.select(range(nsamples))

    # Tokenize and remove empty lines
    process = functools.partial(process, eos=eos_token)

    tokenized_ds = dataset.map(process, remove_columns=['text'], num_proc=16)

    # Writen tokenized_ds to a single file
    doc_len = sum(tokenized_ds['len'])
    doc = np.memmap(fname, dtype=np.uint16, mode='w+', shape=(doc_len,))

    # Write in batches in case of large dataset
    idx = 0
    for batch_idx in tqdm.tqdm(range(batches), desc=f'writing {fname}'):
        batch = tokenized_ds.shard(
            num_shards=batches, index=batch_idx, contiguous=True).with_format('numpy')
        doc_batch = np.concatenate(batch['ids'])
        doc[idx : idx + len(doc_batch)] = doc_batch
        idx += len(doc_batch)
        del batch, doc_batch
    doc.flush()

    return doc_len, eos_token


def create_wikitext103(split='validation', subsample=1.0):
    # Create directory, store file name
    os.makedirs('data/wikitext103', exist_ok=True)
    fdir = 'data/wikitext103'
    if subsample < 1.0:
        subsample_txt = str(subsample).replace('.', '_')
        fname = os.path.join(fdir, f'wikitext103_{split}_{subsample_txt}.npy')
    else:
        fname = os.path.join(fdir, f'wikitext103_{split}.npy')

    # Load dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split=split)

    # Function to modify each sample in dataset
    def process(sample, eos):
        txt = sample['text'][:-2]
        if len(txt) > 0:
            txt
        ids = enc.encode(sample['text'].strip())
        ids.append(eos) 
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Encode dataset and write to file
    enc = tiktoken.get_encoding('gpt2')
    doc_len, eos_token = encode_dataset(
        ds, process, enc, fname=fname, subsample=subsample, batches=1)

    # Store json 
    info = {'doc_len': doc_len,
            'fname': fname, 
            'eos_token': eos_token, 
            'vocab_len': enc.n_vocab}
    info_fname = fname.replace('.npy', '.json')
    with open(info_fname, 'w') as f:
        json.dump(info, f)

    print("Number of tokens in wikitext103 (%s): %d" % (split, doc_len))


def create_wikipediaEn(split='validation', subsample=1.0):
    # val_ds = load_dataset("wikipedia", "20220301.en", split=split)
    pass


if __name__ == '__main__': 
    create_wikitext103('validation')
    create_wikitext103('train')
    # create_wikitext103('train')

