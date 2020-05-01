import logging 
import argparse
import configparser

#from sklearn.model_selection import train_test_split
from model import LSTMAttn
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
    parser.add_argument('-l', '--logger', type=parse_logger, default='', 
                        help='path to logger [stdout]')
    parser.add_argument('-c', '--config', type=str, help='path of the configuration', 
                        default="../../../config/main.conf")
    parser.add_argument('-C', '--cuda_index', type=parse_cuda_index, default='all', 
                        help='which CUDA device to use')
    
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
          check_every=10, 
          lr=1e-4,
          padding_size=3000,
          no_iters=1000,
          fold_size=500,
          no_valid=10,
          no_train=10,
          early_stop=0.05,
          init_max_norm=5,
          max_norm_decay=1,
          mlm_weight=0.5,
          cls_weight=0.5,
          mask_ratio=0.1):
    """
    The training function
    """
    # torch.manual_seed(1)
    
    # loss_function = nn.NLLLoss(ignore_index=0)
    class_loss_function = nn.NLLLoss()
    lm_loss_function = nn.NLLLoss(ignore_index=0)
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
            #model = torch.load(from_checkpoint)
            model.train()
        except:
            logger.error(f"the checkpoint file may not be correct: {from_checkpoint}")
        
    # record the total number of iterations
    total_iters = no_iters * (epochs - curr_epoch)
    
    # process validation set
    sentence_valid = [prepare_sequence(sentence, g_pool['vocab'] , padding_size)
                               for sentence in X_valid]
    sentence_valid_in = torch.stack(sentence_valid)
            
    if g_pool['gpu']:
        target_valid = torch.tensor(np.array(y_valid)).cuda()
    else:
        target_valid = torch.tensor(np.array(y_valid))
        
    idx = 0
    stop_flag = False
    for epoch in range(curr_epoch, epochs):
        logger.info(f"epoch: {epoch}")
        if stop_flag:
            break
        # saving checkpoints
        if epoch % check_every == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f"checkpoints/0_2_cls_0_8_mlm_model.checkpoint_{epoch}")
            #torch.save(model, f"checkpoints/model.checkpoint_{epoch}")
            
        # divide the training data into batchs, or the GPU memory cannot handle that
        for _ in range(no_iters):
            if stop_flag:
                break
            model.zero_grad()
            
            ############################
            # preparing input data
            ############################
            batch_idx = np.random.choice(len(X_train), batch_size, replace=False)
            batch = X_train[batch_idx]
            if g_pool['gpu']:
                target = torch.tensor(np.array([y for y in 
                            y_train[batch_idx]])).cuda()
            else:
                target = torch.tensor(np.array([y for y in 
                            y_train[batch_idx]]))
                
            if not len(target):
                continue
            
            batch_padding_size = max([len(seq) for seq in batch])
            sentence_batch = [prepare_sequence(sentence, g_pool['vocab'], padding_size)
                               for sentence in batch]
            
            # create mask
            token_mask = mask_generator(batch, mask_ratio, padding_size)
            
            sentence_in = torch.stack(sentence_batch)
            if g_pool['gpu']:
                sentence_in = sentence_in.cuda()
            tag_scores, mlm_scores = model(sentence_in, batch_padding_size, token_mask)
            
            # analyze the loss
            cls_loss = class_loss_function(tag_scores, target)
            
            # we only compute cross entropy for the masked: mask==1 part
            # we mask all the other tokens to be 0 and since the loss function 
            # will ignore all the place of 0, it's actually only considering
            # the content within the mask
            sentence_in_mask = sentence_in.masked_fill(token_mask == 1, 0)
            mlm_loss = lm_loss_function(mlm_scores, sentence_in_mask)
            
            loss = (cls_weight * cls_loss + mlm_weight * mlm_loss) / (cls_weight + mlm_weight)
            
            loss_track.append(loss)
            loss.backward()
            # gradient clipping
            # with a decay of the maximum of total norm bound.
            if init_max_norm:
                # when init_max_norm = 0, means we don't need to clip
                remain_rate = float(total_iters - max_norm_decay * idx) / total_iters
                decayed_max_norm = init_max_norm ** remain_rate
                torch.nn.utils.clip_grad_norm(model.parameters(), decayed_max_norm, norm_type=2)
            optimizer.step()
            
            if idx % 100 == 0:
                logger.info(f"iteration no: {idx}/{total_iters}")
                logger.debug(f'batch_padding_size: {batch_padding_size}')
                
                # look at the training accuracy of this batch
                train_cls_acc, train_mlm_acc = test(model, X_train, y_train, test_size=fold_size, 
                                 test_times=no_train, padding_size=padding_size,
                                 random=True, mask_ratio=mask_ratio)
                valid_cls_acc, valid_mlm_acc = test(model, X_valid, y_valid, test_size=fold_size, 
                                 test_times=no_valid, padding_size=padding_size,
                                 random=True, mask_ratio=mask_ratio)
                
                # check norm
                total_norm = 0.0
                for p in model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                logger.info(f"current weighted total loss: {loss}, classifier loss: {cls_loss}, masked language model loss: {mlm_loss}")
                logger.info(f"current training acc: cls: {train_cls_acc:.2%}, mlm: {train_mlm_acc:.2%}") 
                logger.info(f"current validation acc: cls: {valid_cls_acc:.2%}, mlm: {valid_mlm_acc:.2%}") 
                logger.info(f"current model total gradient norm: {total_norm}")   
                
                if total_norm < early_stop and idx > 0.75 * epochs * no_iters:
                    # stop when the gradient is small and done minimum iterations of training
                    logger.info(f"early exit with total_norm: {total_norm}, which is under the threshold: {early_stop}")
                    stop_flag = True
           
            idx += 1
            
    # save the model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f"model/cluster_xlstm_xposenc_expx.model")
    
    
    return loss_track
    
def test(model, X_test, y_test, test_size=500, 
         test_times=10, padding_size=3000, random=False, mask_ratio=0.1):
    cls_acc_list, mlm_acc_list = [], []
    idx_list = np.random.choice(len(X_test) // test_size, test_times, replace=False)
        
    with torch.no_grad():
        for idx in idx_list:
            test_idx = []
            if not random:
                test_idx = np.array(list(range(idx * test_size, (idx + 1) * test_size)))
            else:
                test_idx = np.random.choice(len(X_test), test_size, replace=False)
            X_test_fold = X_test[test_idx]
            token_mask = mask_generator(X_test_fold, mask_ratio, padding_size)
            y_test_fold = y_test[test_idx]
            batch_padding_size = max([len(seq) for seq in X_test_fold])
            X_test_fold = [prepare_sequence(sentence, g_pool['vocab'] , padding_size)
                                   for sentence in X_test_fold]
            X_test_fold = torch.stack(X_test_fold)
            if g_pool['gpu']:
                X_test_fold = X_test_fold.cuda()
            score_pred, mlm_pred = model(X_test_fold, batch_padding_size)
            
            cls_acc_sgl, mlm_acc_sgl = 0, 0
            if score_pred is not None:
                cls_pred_fold = np.array(torch.max(score_pred, 1)[1].tolist())
                cls_acc_sgl = sum(y_test_fold == cls_pred_fold) / len(y_test_fold)
                cls_acc_list.append(cls_acc_sgl)
            if mlm_pred is not None:
                mlm_pred_fold = torch.max(mlm_pred, 1)[1]
                # this is the compare result
                mlm_comp_res = mlm_pred_fold == X_test_fold
                # this is where we care, we will set these position to be 1
                care_part = token_mask == 0
                mlm_masked_res = mlm_comp_res & care_part
                
                mlm_acc_sgl = np.sum(mlm_masked_res.tolist())/np.sum(care_part.tolist())
                mlm_acc_list.append(mlm_acc_sgl)
                
    return np.mean(cls_acc_list or [0]), np.mean(mlm_acc_list or [0])

def run_serial(kwargs):
    config = kwargs['config']
    
    # configuration
    conf, model_conf = load_conf(config)
    cuda_index = kwargs['cuda_index']
    debug = bool(model_conf['Basic']['Debug'])
    
    gpu = bool(int(model_conf['Device']['Gpu']))
    
    from_checkpoint = model_conf['Training']['Reload']
    from_checkpoint = None if from_checkpoint == 'None' else from_checkpoint
    
    clustered_split = bool(int(model_conf['Preprocess']['ClusteredSplit']))
    
    epochs = int(model_conf['Training']['Epochs'])
    no_iters = int(model_conf['Training']['NoIters'])
    early_stop = float(model_conf['Training']['EarlyStop'])
    init_max_norm = float(model_conf['Training']['InitMaxNorm'])
    max_norm_decay = float(model_conf['Training']['MaxNormDecay'])
    
    emb_dim = int(model_conf['Params']['EmbDim'])
    hid_dim = int(model_conf['Params']['HidDim'])
    num_of_folds = int(model_conf['Params']['NumOfFolds'])
    padding_size = int(model_conf['Params']['PaddingSize'])
    batch_size = int(model_conf['Params']['BatchSize'])
    lr = float(model_conf['Params']['Learning_rate'])
    n_lstm = int(model_conf['Params']['NLSTM'])
    n_head = int(model_conf['Params']['NHeaders'])
    n_attn = bool(int(model_conf['Params']['NAttn']))
    need_pos_enc = bool(int(model_conf['Params']['NeedPosEnc']))
    mlm_weight = float(model_conf['Params']['MlmWeight'])
    cls_weight = float(model_conf['Params']['ClsWeight'])
    mask_ratio = float(model_conf['Params']['MaskRatio'])
    
    # testing parameters
    fold_size = int(model_conf['Test']['FoldSize'])
    no_test = int(model_conf['Test']['NoTest'])
    no_valid = int(model_conf['Test']['NoValid'])
    no_train = int(model_conf['Test']['NoTrain'])
    
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
    
    X_train, X_test, X_dev, y_train, y_test, y_dev = load_data(
        conf, logger, g_pool, clustered_split=clustered_split)
    
    logger.debug("finish loading data")
    
    # get model
    model = LSTMAttn(embedding_dim=emb_dim, 
                     hidden_dim=hid_dim, 
                     seq_len=padding_size, 
                     vocab=g_pool['vocab'],
                     tagset_size=len(g_pool['fams']),
                     n_lstm=n_lstm, 
                     n_head=n_head,
                     n_attn=n_attn,
                     need_pos_enc=need_pos_enc,
                     is_gpu=gpu,
                     mask_ratio=mask_ratio)
    
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
                       X_dev,
                       y_dev,
                       model, 
                       epochs, 
                       batch_size, 
                       logger,
                       from_checkpoint=from_checkpoint,
                       lr = lr,
                       padding_size=padding_size,
                       fold_size=fold_size,
                       no_iters=no_iters,
                       no_train=no_train,
                       no_valid=no_valid,
                       early_stop=early_stop,
                       init_max_norm=init_max_norm,
                       max_norm_decay=max_norm_decay,
                       mlm_weight=mlm_weight,
                       cls_weight=cls_weight,
                       mask_ratio=mask_ratio)
    logger.debug("end training")
    
    # testing the result
    # because of the limitation of the GPU memory
    # have to test the result for multiple times
    cls_acc, mlm_acc = test(model, X_test, y_test, test_size=fold_size, 
               test_times=no_test, padding_size=padding_size,
               random=False)
    logger.info(f"The final accuracy is: cls: {cls_acc:.2%}, mlm: {mlm_acc:.2%}")
    return

if __name__ == '__main__':
    kwargs = parse_args()
    run_serial(kwargs)