import tiktoken
import json
import tqdm
import os
import numpy as np
from datasets import load_dataset


def encode_dataset(dataset, enc, fname, subsample=1.0, batches=1):
    newline = enc.encode('\n\n')[0]

    # Function to modify each sample in dataset
    def process(sample):
            ids = enc.encode(sample['text'].strip())
            ids.append(newline) 
            out = {'ids': ids, 'len': len(ids)}
            return out

    # Subsample dataset
    if subsample < 1.0:
        nsamples = int(len(dataset) * subsample)
        dataset = dataset.select(range(nsamples))

    import ipdb; ipdb.set_trace()

    # Tokenize and remove empty lines
    tokenized_ds = dataset.map(process, remove_columns=['text'], num_proc=16)
    tokenized_ds = tokenized_ds.filter(lambda sample: len(sample['ids']) > 1, num_proc=8)


    doc_len = sum(tokenized_ds['len'])

    # Writen tokenized_ds to a single file
    doc = np.memmap(fname, dtype=np.uint16, mode='w+', shape=(doc_len,))

    idx = 0
    for batch_idx in tqdm.tqdm(range(batches), desc=f'writing {fname}'):
        batch = tokenized_ds.shard(
            num_shards=batches, index=batch_idx, contiguous=True).with_format('numpy')
        doc_batch = np.concatenate(batch['ids'])
        doc[idx : idx + len(doc_batch)] = doc_batch
        idx += len(doc_batch)
        del batch, doc_batch
    doc.flush()

    return doc_len, newline


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

    # Encode dataset and write to file
    enc = tiktoken.get_encoding('gpt2')
    doc_len, eos_token = encode_dataset(
        ds, enc, fname=fname, subsample=subsample, batches=1)

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
    # create_wikitext103('train')

