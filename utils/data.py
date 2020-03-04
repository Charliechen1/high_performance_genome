import logging 
import argparse
import configparser
import torch

from sklearn.model_selection import train_test_split
import sys, os
import pandas as pd
import json
import numpy as np
    
def prepare_sequence(seq, vocab, padding):
    """
    function to process the data, padding them
    TODO later will move to specific preprocessing part
    """
    res = ['<PAD>'] * padding
    res[:min(padding, len(seq))] = seq[:min(padding, len(seq))]
    # use 0 for padding
    idxs = [vocab[w] for w in res]
    return torch.tensor(idxs, dtype=torch.long)

def load_data(conf, logger, g_pool):
    """
    function to load data
    """
    
    model_conf = configparser.ConfigParser()
    model_conf.read(conf['path']['model'])
    
    test_size = float(model_conf['Training']['TestRate'])
    sample_every = int(model_conf['Training']["SampleEvery"])
    
    data_partitions_dirpath = conf['path']['data_part']
    print('Available dataset partitions: ', os.listdir(data_partitions_dirpath))

    def read_all_shards(partition='dev', data_dir=data_partitions_dirpath):
        shards = []
        for fn in os.listdir(os.path.join(data_dir, partition)):
            with open(os.path.join(data_dir, partition, fn)) as f:
                shards.append(pd.read_csv(f, index_col=None))
        return pd.concat(shards)

    test = read_all_shards('test')
    dev = read_all_shards('dev')
    train = read_all_shards('train')

    partitions = {'test': test, 'dev': dev, 'train': train}
    for name, df in partitions.items():
        logger.info('Dataset partition "%s" has %d sequences' % (name, len(df)))

    # load vocab
    vocab_path = conf['path']['vocab']
    with open(vocab_path, 'r') as of:
        vocab = json.load(of)
    g_pool['vocab'] = vocab
    fams = np.array(train["family_id"].value_counts().index)[::sample_every]
    g_pool['fams'] = fams
    partition = train[train["family_id"].isin(fams)]
    max_len = int(model_conf['Preprocess']['MaxLen'])
    X = partition['aligned_sequence'].values
    y = partition['family_id'].values
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=test_size, random_state=45)
    fam_vocab = {fam: idx for idx, fam in enumerate(fams)}
    g_pool['fam_vocab'] = fam_vocab
    
    y_train = np.array([fam_vocab[y] for y in y_train_raw])
    y_test = np.array([fam_vocab[y] for y in y_test_raw])
    return X_train_raw, X_test_raw, y_train, y_test