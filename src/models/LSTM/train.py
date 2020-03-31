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
    parser.add_argument('-l', '--logger', type=parse_logger, default='', help='path to logger [stdout]')
    parser.add_argument('-c', '--config', type=str, help='path of the configuration', default="../../../config/main.conf")
    parser.add_argument('-C', '--cuda_index', type=parse_cuda_index, default='all', help='which CUDA device to use')
    
    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args(argv)
    ret = vars(args)
    return ret
        
def train(X_train, 
          y_train,
          X_valid,
          y_valid,
          model, 
          epochs, 
          batch_size, 
          logger, 
          from_checkpoint=None, 
          check_every=1, 
          lr=1e-4,
          padding_size=3000):
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
        
    # record the total number of iterations
    total_iters = math.ceil(len(X_train) / batch_size) * epochs
    batch_no = math.ceil(len(X_train) / batch_size)
    
    # process validation set
    sentence_valid = [prepare_sequence(sentence, g_pool['vocab'] , padding_size)
                               for sentence in X_valid]
    sentence_valid_in = torch.stack(sentence_valid)
            
    if g_pool['gpu']:
        target_valid = torch.tensor(np.array(y_valid)).cuda()
    else:
        target_valid = torch.tensor(np.array(y_valid))
        
    idx = 0
    for epoch in range(curr_epoch, epochs):
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
        for batch_idx in range(batch_no):
            model.zero_grad()
            batch = X_train[batch_size * batch_idx:batch_size * (batch_idx + 1)]
            if g_pool['gpu']:
                target = torch.tensor(np.array([y for y in 
                            y_train[batch_size * batch_idx:batch_size * (batch_idx + 1)]])).cuda()
            else:
                target = torch.tensor(np.array([y for y in 
                            y_train[batch_size * batch_idx:batch_size * (batch_idx + 1)]]))
                
            if not len(target):
                continue
            sentence_batch = [prepare_sequence(sentence, g_pool['vocab'], padding_size)
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
            
    # save the model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f"model/model_LSTM.final_model")
    
    return loss_track
            
def run_serial(kwargs):
    config = kwargs['config']
    
    # configuration
    conf, model_conf = load_conf(config)
    print(config)
    print(conf, model_conf)
    cuda_index = kwargs['cuda_index']
    debug = bool(model_conf['Basic']['Debug'])
    
    gpu = bool(int(model_conf['Device']['Gpu']))
    
    from_checkpoint = model_conf['Training']['Reload']
    from_checkpoint = None if from_checkpoint == 'None' else from_checkpoint
    
    epochs = int(model_conf['Training']['Epochs'])
    
    emb_dim = int(model_conf['Params']['EmbDim'])
    hid_dim = int(model_conf['Params']['HidDim'])
    num_of_folds = int(model_conf['Params']['NumOfFolds'])
    padding_size = int(model_conf['Params']['PaddingSize'])
    batch_size = int(model_conf['Params']['BatchSize'])
    lr = float(model_conf['Params']['Learning_rate'])
    n_lstm = int(model_conf['Params']['NLayers'])
    n_head = int(model_conf['Params']['NHeaders'])
    need_attn = bool(int(model_conf['Params']['NeedAttn']))
    
    g_pool['gpu'] = gpu
    
    # prepare logging
    logger = kwargs["logger"]
    
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)
    #logger.info("The arguments are: %s", kwargs)
    logger.info("The configuration are: %s", json.dumps(model_conf._sections))
    
    # load data
    # data should be load before model
    # as there are vocab and fams
    logger.debug("start loading data")
    X_train, X_test, y_train, y_test = load_data(conf, logger, g_pool)
    
    # because the GPU Mem is not able to load all the text data
    valid_size = 500
    valid_idx = np.random.choice(len(X_test), valid_size)
    X_valid = X_test[valid_idx]
    y_valid = y_test[valid_idx]
    
    test_size = 500
    text_idx = np.random.choice(len(X_test), test_size)
    X_test = X_test[text_idx]
    y_test = y_test[text_idx]
    
    logger.debug("finish loading data")
    
    # get model
    model = LSTMTagger(embedding_dim=emb_dim, 
                     hidden_dim=hid_dim, 
                     seq_len=padding_size, 
                     vocab_size=len(g_pool['vocab']),
                     tagset_size=len(g_pool['fams']),
                     n_lstm=n_lstm, 
                     n_head=n_head,
                     need_attn=need_attn)
    
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
                       X_valid,
                       y_valid,
                       model, 
                       epochs, 
                       batch_size, 
                       logger,
                       from_checkpoint=from_checkpoint,
                       lr = lr,
                       padding_size=padding_size)
    logger.debug("end training")
    
    # testing the result
    X_test = [prepare_sequence(sentence, g_pool['vocab'] , padding_size)
                           for sentence in X_test]
    X_test = torch.stack(X_test)
    score_pred = model(X_test)
    y_pred = np.array(torch.max(score_pred, 1)[1].tolist())
    acc = sum(y_test == y_pred) / len(y_test)
    logger.info(f"The final accuracy is {acc:.2%}")

if __name__ == '__main__':
    kwargs = parse_args()
    run_serial(kwargs)