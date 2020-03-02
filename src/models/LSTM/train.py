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
    parser.add_argument('-b', '--batch_size', type=int, help='divide the training data into batch', default=200)
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='run in debug mode i.e. only run two batches')
    parser.add_argument('-l', '--logger', type=parse_logger, default='', help='path to logger [stdout]')
    parser.add_argument('-g', '--gpu', action='store_true', default=False, help='use GPU')
    parser.add_argument('-c', '--config', type=str, help='path of the configuration', default="../../../config/main.conf")
    parser.add_argument('-C', '--cuda_index', type=parse_cuda_index, default='all', help='which CUDA device to use')
    parser.add_argument('-r', '--reload', type=str, default='', help='reload from the checkpoint for training')
    parser.add_argument('-L', '--learning_rate', type=float, default=1e-4, help='learning rate of the model')
    
    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args(argv)
    ret = vars(args)
    return ret
        
def train(X_train, y_train, model, epoches, batch_size, logger, from_checkpoint=None, check_every=3, lr=1e-4):
    """
    The training function
    """
    torch.manual_seed(1)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_track = []
    
    curr_epoch = 0
    # load model from checkpoint
    if from_checkpoint is not None and from_checkpoint != "":
        try:
            # from the checkpoint filename, we can know the epoch the model is trained
            checkpoint = torch.load(from_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            curr_epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            model.eval()
        except:
            logger.error(f"the checkpoint file may not be correct: {from_checkpoint}")
            model.zero_grad()
    else:
        model.zero_grad()
    
    # record the total number of iterations
    total_iters = math.ceil(len(X_train) / batch_size) * epoches
    batch_no = math.ceil(len(X_train) / batch_size)
    
    # the last batch will be used as validation
    X_valid = X_train[batch_size * (batch_no - 1):batch_size * batch_no]
    sentence_valid = [prepare_sequence(sentence, g_pool['vocab'] , kwargs['padding_size'])
                               for sentence in X_valid]
    sentence_valid_in = torch.stack(sentence_valid)
            
    y_valid = y_train[batch_size * (batch_no - 1):batch_size * batch_no]
    if g_pool['gpu']:
        target_valid = torch.tensor(np.array(y_valid)).cuda()
    else:
        target_valid = torch.tensor(np.array(y_valid))
        
    idx = 0
    for epoch in range(curr_epoch, epoches):
        logger.info(f"epoch: {epoch}")
        
        # saving checkpoints
        if epoch % check_every == 0 and epoch > 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f"model/model_LSTM.checkpoint_{epoch}")
            
        # divide the training data into batchs, or the GPU memory cannot handle that
        for batch_idx in range(batch_no - 1):
            batch = X_train[batch_size * batch_idx:batch_size * (batch_idx + 1)]
            if g_pool['gpu']:
                target = torch.tensor(np.array([y for y in 
                            y_train[batch_size * batch_idx:batch_size * (batch_idx + 1)]])).cuda()
            else:
                target = torch.tensor(np.array([y for y in 
                            y_train[batch_size * batch_idx:batch_size * (batch_idx + 1)]]))
                
            if not len(target):
                continue
            sentence_batch = [prepare_sequence(sentence, g_pool['vocab'] , kwargs['padding_size'])
                               for sentence in batch]
            sentence_in = torch.stack(sentence_batch)
            tag_scores = model(sentence_in)

            loss = loss_function(tag_scores, target)
            loss_track.append(loss)
            if idx % 100 == 0:
                logger.info(f"iteration no: {idx}/{total_iters}")
                
                # look at the training accuracy of this batch
                y_pred = torch.max(tag_scores, 1)[1]
                training_correct = (target.eq(y_pred.long())).sum()
                training_acc = training_correct.to(dtype=torch.float) / float(len(target))
                
                y_pred_valid = torch.max(model(sentence_valid_in), 1)[1]
                valid_correct = (target_valid.eq(y_pred_valid.long())).sum()
                valid_acc = valid_correct.to(dtype=torch.float) / float(len(y_valid))
                
                logger.info(f"current loss: {loss}")
                logger.info(f"current training acc: {training_acc:.2%}") 
                logger.info(f"current validation acc: {valid_acc:.2%}") 
                
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
    batch_size = kwargs['batch_size']
    from_checkpoint = kwargs['reload']
    lr = kwargs['learning_rate']
    # configuration
    conf, model_conf = load_conf(config)
    
    g_pool['gpu'] = gpu
    
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
    
    # because the GPU Mem is not able to load all the text data
    test_size = 2000
    X_test = X_test[:test_size]
    y_test = y_test[:test_size]
    
    logger.debug("finish loading data")
    
    # get model
    model = LSTMTagger(kwargs["emb_dim"], 
                       kwargs["hid_dim"],
                       kwargs["padding_size"], 
                       len(g_pool['vocab']), 
                       len(g_pool['fams']), 
                       kwargs["num_of_folds"],
                      )
    
    # check device
    if gpu:
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
    loss_track = train(X_train, 
                       y_train, 
                       model, 
                       epoches, 
                       batch_size, 
                       logger,
                       from_checkpoint=from_checkpoint,
                       lr = lr)
    logger.debug("end training")
    
    # save the model
    torch.save(model, f"model/model_LSTM.final_model")
    
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