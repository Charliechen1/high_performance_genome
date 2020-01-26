import logging 
import argparse
import configparser

from sklearn.model_selection import train_test_split
from model import LSTMTagger
import torch
import sys, os
import pandas as pd
import json
import numpy as np

import torch.nn as nn
import torch.optim as optim

# define global variables
g_pool = {}
def parse_logger(string):
    if not string:
        ret = logging.getLogger('stdout')
        hdlr = logging.StreamHandler(sys.stdout)
    else:
        ret = logging.getLogger(string)
        hdlr = logging.FileHandler(string)
    ret.setLevel(logging.INFO)
    ret.addHandler(hdlr)
    hdlr.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    return ret

def parse_args(argv=None):
    """
    Parse arguments for training executable
    """
    if argv is None:
        argv = sys.argv[1:]
       
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample_rate', type=int, help='rate of sampling', default=1)
    parser.add_argument('-e', '--emb_dim', type=int, help='the embedding dimension', default=30)
    parser.add_argument('-H', '--hid_dim', type=int, help='the hidden layer dimension', default=20)
    parser.add_argument('-b', '--batch_size', type=int, help='the size of the batch', default=128)
    parser.add_argument('-p', '--padding_size', type=int, help='the size of the padding', default=300)
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='run in debug mode i.e. only run two batches')
    parser.add_argument('-l', '--logger', type=parse_logger, default='', help='path to logger [stdout]')
    parser.add_argument('-g', '--gpu', action='store_true', default=False, help='use GPU')
    parser.add_argument('-c', '--config', type=str, help='path of the configuration', default="../../../config/main.conf")
    parser.add_argument('-C', '--cuda_index', type=parse_cuda_index, default='all', help='which CUDA device to use')
    
    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args(argv)
    ret = vars(args)
    return ret

def _parse_cuda_index_helper(s):
    try:
        i = int(s)
        if i > torch.cuda.device_count() or i < 0:
            raise ValueError(s)
        return i
    except :
        devices = str(np.arange(torch.cuda.device_count()))
        raise argparse.ArgumentTypeError(f'{s} is not a valid CUDA index. Please choose from {devices}')
        
        
def parse_cuda_index(string):
    if string == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        if ',' in string:
            return [_parse_cuda_index_helper(_) for _ in string.split(',')]
        else:
            return _parse_cuda_index_helper(string)
        
        
def train(X_train, y_train, model, epoches=10):
    """
    The training function
    """
    torch.manual_seed(1)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)

    batched_training_data = []
    loss_track = []
    # separate data to each batch
    for idx in range(len(X_train) // kwargs["batch_size"] + 1):
        X_batch = X_train[kwargs["batch_size"] * idx:kwargs["batch_size"] * (idx + 1)]
        y_data = torch.tensor(np.array([y for y in 
                    y_train[kwargs["batch_size"] * idx:kwargs["batch_size"] * (idx + 1)]]))
        if not len(y_data):
            continue
        batched_training_data.append((X_batch, y_data))

    model.zero_grad()
    for epoch in range(epoches):
        logger.debug("epoch: %d" % epoch)
        idx = 0
        for batch, target in batched_training_data:
            sentence_batch = [prepare_sequence(sentence, g_pool['vocab'] , kwargs['padding_size'])
                               for sentence in batch]
            sentence_in = torch.stack(sentence_batch)
            tag_scores = model(sentence_in)

            #labels = torch.max(target, 1)[1]
            #loss = loss_function(tag_scores, labels.long())
            loss = loss_function(tag_scores, target)
            loss_track.append(loss)
            if idx % 100 == 0:
                logger.debug("batch no: %d" % idx)
                logger.debug("loss: %f" % loss)
            loss.backward()
            optimizer.step()
            idx += 1
            
def load_conf(config_path):
    """
    function to load the global configuration
    """
    global conf, model_conf
    conf = configparser.ConfigParser()
    conf.read(config_path)
    print(conf)
            
    model_conf = configparser.ConfigParser()
    model_conf.read(conf['path']['model'])
    
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

def load_data():
    """
    function to load data
    """
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
    fams = np.array(train["family_id"].value_counts().index)[::kwargs["sample_rate"]]
    g_pool['fams'] = fams
    partition = train[train["family_id"].isin(fams)]
    max_len = int(model_conf['Preprocess']['MaxLen'])
    X = partition['aligned_sequence'].values
    y = partition['family_id'].values
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.25, random_state=41)
    fam_vocab = {fam: idx for idx, fam in enumerate(fams)}
    
    y_train = np.array([fam_vocab[y] for y in y_train_raw])
    y_test = np.array([fam_vocab[y] for y in y_test_raw])
    return X_train_raw, X_test_raw, y_train, y_test

def run_serial():
    config = kwargs['config']
    cuda_index = kwargs['cuda_index']
    debug = kwargs['debug']
    gpu = kwargs['gpu']
    cuda_index = kwargs['cuda_index']
    
    # configuration
    load_conf(config)
    
    # prepare logging
    global logger
    logger = kwargs["logger"]
    
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    if kwargs["debug"]:
        logger.setLevel(logging.DEBUG)
    logger.info("The arguments are: %s", kwargs)
    
    # load data
    # data should be load before model
    # as there are vocab and fams
    logger.debug("start loading data")
    X_train, X_test, y_train, y_test = load_data()
    logger.debug("finish loading data")
    
    # get model
    model = LSTMTagger(kwargs["emb_dim"], 
                       kwargs["hid_dim"], 
                       len(g_pool['vocab']), 
                       len(g_pool['fams']), 
                       kwargs["batch_size"])
    
    # check device
    if gpu:
        print(cuda_index)
        to_index = cuda_index
        if isinstance(cuda_index, list):
            to_index = cuda_index[0]
            model = nn.DataParallel(model, device_ids=cuda_index)
            logger.info(f'running on GPUs {cuda_index}')
        device = torch.device(f"cuda:{to_index}")
        logger.info(f'sending data to CUDA device {str(device)}')
        model.to(device)
    
    
    # train model
    logger.debug("start training")
    train(X_train, y_train, model)
    logger.debug("end training")
    
    # testing the result
    X_test = [prepare_sequence(sentence, g_pool['vocab'] , kwargs["padding_size"])
                           for sentence in X_test_raw]
    X_test = torch.stack(X_test)
    score_pred = model(X_test)
    y_pred = np.array(torch.max(score_pred, 1)[1].tolist())
    y_test = np.array([fam_vocab[fam] for fam in y_test_raw])
    acc = sum(y_test == y_pred) / len(y_test)
    logger.info("The accuracy is %d" % acc)

if __name__ == '__main__':
    global kwargs
    kwargs = parse_args()
    run_serial()