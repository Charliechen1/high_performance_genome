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
import math

import torch.nn as nn
import torch.optim as optim

sys.path.append("../../../utils/")

from data import *
from args import *
from conf import *

# define global variables
g_pool = {}

def parse_args(argv=None):
    """
    Parse arguments for training executable
    """
    if argv is None:
        argv = sys.argv[1:]
       
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample_rate', type=int, help='rate of sampling', default=1)
    parser.add_argument('-e', '--emb_dim', type=int, help='the embedding dimension', default=30)
    parser.add_argument('-E', '--epoches', type=int, help='number of epoches', default=10)
    parser.add_argument('-H', '--hid_dim', type=int, help='the hidden layer dimension', default=20)
    parser.add_argument('-f', '--num_of_folds', type=int, help='number of folds', default=10)
    parser.add_argument('-p', '--padding_size', type=int, help='the size of the padding', default=300)
    parser.add_argument('-P', '--part_size', type=int, help='divide the training data into parts', default=200)
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
        
def train(X_train, y_train, model, epoches, part_size, logger):
    """
    The training function
    """
    torch.manual_seed(1)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    loss_track = []

    model.zero_grad()
    
    # record the total number of iterations
    total_iters = math.ceil(len(X_train) / part_size) * epoches
    
    idx = 0
    for epoch in range(epoches):
        logger.info(f"epoch: {epoch}")
        
        # divide the training data into parts, or the GPU memory cannot handle that
        for part_idx in range(len(X_train) // part_size + 1):
            part = X_train[part_size * part_idx:part_size * (part_idx + 1)]
            target = torch.tensor(np.array([y for y in 
                        y_train[part_size * part_idx:part_size * (part_idx + 1)]])).cuda()
            if not len(target):
                continue
            sentence_part = [prepare_sequence(sentence, g_pool['vocab'] , kwargs['padding_size'])
                               for sentence in part]
            sentence_in = torch.stack(sentence_part)
            tag_scores = model(sentence_in)

            loss = loss_function(tag_scores, target)
            loss_track.append(loss)
            if idx % 100 == 0:
                logger.info(f"iteration no: {idx}/{total_iters}")
                
                # look at the training accuracy of this part
                y_pred = torch.max(tag_scores, 1)[1]
                correct = (target.eq(y_pred.long())).sum()
                acc = correct.to(dtype=torch.float) / float(len(target))
                
                logger.info(f"current loss: {loss}")
                logger.info(f"current training correct: {correct} of {len(target)} acc: {acc:.2%}") 
                
            loss.backward()
            optimizer.step()
            idx += 1
        
    return loss_track
            
def run_serial(kwargs):
    config = kwargs['config']
    cuda_index = kwargs['cuda_index']
    debug = kwargs['debug']
    gpu = kwargs['gpu']
    epoches = kwargs['epoches']
    num_of_folds = kwargs['num_of_folds']
    part_size = kwargs['part_size']
    # configuration
    conf, model_conf = load_conf(config)
    
    # prepare logging
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
    X_train, X_test, y_train, y_test = load_data(conf, logger, g_pool, kwargs)
    logger.debug("finish loading data")
    
    # get model
    model = LSTMTagger(kwargs["emb_dim"], 
                       kwargs["hid_dim"], 
                       len(g_pool['vocab']), 
                       len(g_pool['fams']), 
                       kwargs["num_of_folds"],
                      )
    
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
    loss_track = train(X_train, y_train, model, epoches, part_size, logger)
    logger.debug("end training")
    
    # testing the result
    X_test = [prepare_sequence(sentence, g_pool['vocab'] , kwargs["padding_size"])
                           for sentence in X_test]
    X_test = torch.stack(X_test)
    score_pred = model(X_test)
    y_pred = np.array(torch.max(score_pred, 1)[1].tolist())
    acc = sum(y_test == y_pred) / len(y_test)
    logger.info(f"The final accuracy is {acc}")

if __name__ == '__main__':
    kwargs = parse_args()
    run_serial(kwargs)