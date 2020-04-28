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

def load_data(conf, logger, g_pool, clustered_split=False):
    """
    function to load data
    """
    model_conf = configparser.ConfigParser()
    model_conf.read(conf['path']['model'])
    
    test_size = float(model_conf['Training']['TestRate'])
    sample_every = int(model_conf['Training']["SampleEvery"])
    
    data_partitions_dirpath = conf['path']['data_part']
    if clustered_split:
        data_partitions_dirpath = conf['path']['clustered_split']
    
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
    
    train_partition = train[train["family_id"].isin(fams)]
    X_train_raw = train_partition['aligned_sequence'].values
    y_train_raw = train_partition['family_id'].values
    
    test_partition = test[test["family_id"].isin(fams)]
    X_test_raw = test_partition['aligned_sequence'].values
    y_test_raw = test_partition['family_id'].values
    
    dev_partition = dev[dev["family_id"].isin(fams)]
    X_dev_raw = dev_partition['aligned_sequence'].values
    y_dev_raw = dev_partition['family_id'].values
    
    fam_vocab = {fam: idx for idx, fam in enumerate(fams)}
    g_pool['fam_vocab'] = fam_vocab
    
    y_train = np.array([fam_vocab[y] for y in y_train_raw])
    y_test = np.array([fam_vocab[y] for y in y_test_raw])
    y_dev = np.array([fam_vocab[y] for y in y_dev_raw])
    
    X_train_len = [len(seq) for seq in X_train_raw]
    X_test_len = [len(seq) for seq in X_test_raw]
    X_dev_len = [len(seq) for seq in X_dev_raw]
    
    logger.info(f'X train len: avg: {sum(X_train_len)/len(X_train_len)}, max: {max(X_train_len)}, min: {min(X_train_len)}')
    logger.info(f'X test len: avg: {sum(X_test_len)/len(X_test_len)}, max: {max(X_test_len)}, min: {min(X_test_len)}')
    logger.info(f'X dev len: avg: {sum(X_dev_len)/len(X_dev_len)}, max: {max(X_dev_len)}, min: {min(X_dev_len)}')
    
    return X_train_raw, X_test_raw, X_dev_raw, y_train, y_test, y_dev

if __name__ == '__main__':
    # testing code
    X_train_raw, X_test_raw, X_dev_raw, y_train, y_test, y_dev = load_data(conf, logger, g_pool, clustered_split=False)
    
    X_train_len = [len(seq) for seq in X_train_raw]
    X_test_len = [len(seq) for seq in X_test_raw]
    X_dev_len = [len(seq) for seq in X_dev_raw]
    
    plt.plot(X_train_len)
    plt.plot(X_test_len)
    plt.plot(X_dev_len)