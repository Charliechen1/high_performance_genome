import pandas as pd
import numpy as np
import tensorflow as tf
import os
import configparser
import json

from keras.utils import to_categorical

from scipy import sparse
import json

def read_all_shards(partition='dev', data_dir=data_partitions_dirpath):
    shards = []
    for fn in os.listdir(os.path.join(data_dir, partition)):
        with open(os.path.join(data_dir, partition, fn)) as f:
            shards.append(pd.read_csv(f, index_col=None))
    return pd.concat(shards)

def one_hot_encoding(X, vocab, max_len):
    row_index = np.arange((max_len))
    num = 0
    X_res = None
    for x in X:
        num += 1
        if (num % 100000 == 0):
            print("No %d in X part" % num)
        col_index = [vocab.get(ch, 0) for ch in x]
        xlen = len(col_index)
        data = np.array([1] * min(max_len, xlen) + [0] * max(max_len - xlen, 0))
        # chop the indices
        col_index = col_index[:max_len] + [0] * max(max_len - xlen, 0)
        new_sparse = sparse.coo_matrix((data, (row_index, col_index)), shape=(max_len, 26))
        if X_res is None:
            X_res = new_sparse
        else:
            sparse.hstack((X_res, new_sparse))
    return X_res

def parse_and_save_dense_mat(X, path, to_sparse=True):
    if to_sparse:
        sparse.save_npz(path, X)
    else:
        with open(path, 'w') as of:
            for x in X:
                x_ = x.toarray().tolist()
                of.write(json.dumps(x_) + '\n')
                
if __name__ == '__main__':
    config_path = "../config/main.conf"
    conf = configparser.ConfigParser()
    conf.read(config_path)
    
    data_partitions_dirpath = conf['path']['data_part']
    print('Available dataset partitions: ', os.listdir(data_partitions_dirpath))
    
    test = read_all_shards('test')
    dev = read_all_shards('dev')
    train = read_all_shards('train')

    partitions = {'test': test, 'dev': dev, 'train': train}
    for name, df in partitions.items():
        print('Dataset partition "%s" has %d sequences' % (name, len(df)))
        
    vocab = None
    vocab_path = conf['path']['vocab']
    with open(vocab_path, 'r') as of:
        vocab = json.load(of)
    
    prepro_conf = configparser.ConfigParser()
    prepro_conf.read(conf['path']['preprocessing'])
    max_len = prepro_conf['OneHot']['MaxLen']
    X_train_raw = train['aligned_sequence'].values
    X_test_raw = test['aligned_sequence'].values
    X_train = one_hot_encoding(X_train_raw, vocab)
    X_test = one_hot_encoding(X_test_raw, vocab)
    
    y_train_raw, y_test_raw = train['family_id'].values, test['family_id'].values
    all_fam = {fam: idx for idx, fam in enumerate(list(set(y_train_raw)))}
    y_train_raw = np.array([all_fam.get(fam, len(all_fam)) for fam in y_train_raw])
    y_test_raw = np.array([all_fam.get(fam, len(all_fam)) for fam in y_test_raw])
    
    y_train = one_hot_encoding([y_train_raw], all_fam, max_len)
    y_test = one_hot_encoding([y_test_raw], all_fam, max_len)

    parse_and_save_dense_mat(X_train, conf['path']['x_train'])
    parse_and_save_dense_mat(X_test, conf['path']['x_test'])
    parse_and_save_dense_mat(y_train, conf['path']['y_train'])
    parse_and_save_dense_mat(y_test, conf['path']['y_test'])