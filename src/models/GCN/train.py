import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from model import GCN
import pickle
import configparser
import logging
import sys
import os
import json
import data
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def split_data(data_dict, aa_vocab, go_type = 'mf'):
    
    
    if go_type == 'mf':
        X = data_dict['X_mf'] 
        y = data_dict['y_mf']  
        vocab = data_dict['mf_vocab']
        
    if go_type == 'bp':
        X = data_dict['X_bp'] 
        y = data_dict['y_bp']  
        vocab = data_dict['bp_vocab']
        
    if go_type == 'cc':
        X = data_dict['X_cc'] 
        y = data_dict['y_cc']  
        vocab = data_dict['cc_vocab']
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.125, random_state=41)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=41)
    dataset_train = []
    for data, raw_labels in zip(X_train, y_train):
        x = prepare_sequence(data[0], aa_vocab, 3000).float()
        edge_index = from_scipy_sparse_matrix(data[1])[0]
        labels = torch.tensor([vocab[y] for y in raw_labels])
        targets = torch.zeros((len(vocab)))
        targets[labels] = 1
        targets = torch.tensor(targets)
        dataset_train.append(Data(x = x, edge_index = edge_index, y = targets))
        
    dataset_val = []
    for data, raw_labels in zip(X_val, y_val):
        x = prepare_sequence(data[0], aa_vocab, 3000).float()
        edge_index = from_scipy_sparse_matrix(data[1])[0]
        labels = torch.tensor([vocab[y] for y in raw_labels])
        targets = torch.zeros((len(vocab)))
        targets[labels] = 1
        targets = torch.tensor(targets)
        dataset_val.append(Data(x = x, edge_index = edge_index, y = targets))
        
    dataset_test = []
    for data, raw_labels in zip(X_test, y_test):
        x = prepare_sequence(data[0], aa_vocab, 3000).float()
        edge_index = from_scipy_sparse_matrix(data[1])[0]
        labels = torch.tensor([vocab[y] for y in raw_labels])
        targets = torch.zeros((len(vocab)))
        targets[labels] = 1
        targets = torch.tensor(targets)
        dataset_val.append(Data(x = x, edge_index = edge_index, y = targets))
        
        
    return dataset_train, dataset_val, dataset_test, len(vocab)

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


def train(logger, model, dataset_train, dataset_val, target_dim, lr=0.00005, num_epoch = 10):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()
    train_loader = DataLoader(dataset_train, shuffle = True, batch_size=64)
    model.train()
    logger.debug('')
    for epoch in range(num_epoch):
        for iteration, data in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.float().reshape(-1, target_dim))
            loss.backward()
            optimizer.step()
            if iteration % 40 == 0:
                logger.debug(f'epoch {epoch + 1}, iteration {iteration}')
                logger.debug(f'loss = {loss}')
                pred = (out > 0.5).float()
                label = data.y.float().reshape(-1, target_dim)
                logger.debug(f'precision = {((label * pred).sum()/label.sum()).item()}')
                logger.debug(f'recall = {((label * pred).sum()/pred.sum()).item()}')
                logger.debug(f'total # of go terms = {label.sum().item()}')
                logger.debug(f'predicted # of go terms = {pred.sum().item()}')
                logger.debug('')
                
def evaluate(model, dataset_train, dataset_val, target_dim):
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(dataset_train, batch_size=64)
    val_loader = DataLoader(dataset_val, batch_size = 64)
    with torch.no_grad():
        train_logits = {}
        val_logits = {}
        train_labels = {}
        val_labels = {}
        train_precisions = []
        train_recalls = []
        val_precisions = []
        val_recalls = []
        for thresh in torch.arange(0.01, .99, .01):
            train_precision = 0.0
            train_recall = 0.0
            train_num_batches = 0.0
            for idx, data in enumerate(train_loader):
                data = data.to(device)
                if idx in train_logits:
                    pred = (train_logits[idx] > thresh).float()
                    label = train_labels[idx]
                else:
                    logits = model(data)
                    train_logits[idx] = logits
                    pred = (logits > thresh).float()
                    label = data.y.float().reshape(-1, target_dim)
                    train_labels[idx] = label
                train_precision += ((label * pred).sum()/label.sum()).item()
                train_recall += ((label * pred).sum()/pred.sum()).item()
                train_num_batches += 1
            #print('precision = ', train_precision/train_num_batches)
            #print('recall = ', train_recall/train_num_batches)
            train_precisions.append(train_precision/train_num_batches)
            train_recalls.append(train_recall/train_num_batches)

            val_precision = 0.0
            val_recall = 0.0
            val_num_batches = 0.0

            for data in val_loader:
                data = data.to(device) 
                if idx in val_logits:
                    pred = (val_logits[idx] > thresh).float()
                    label = val_labels[idx]
                else:
                    logits = model(data)
                    val_logits[idx] = logits
                    pred = (logits > thresh).float()
                    label = data.y.float().reshape(-1, target_dim)
                    val_labels[idx] = label
                if label.sum().item() != 0:
                    val_precision += ((label * pred).sum()/label.sum()).item()
                if pred.sum().item() != 0:
                    val_recall += ((label * pred).sum()/pred.sum()).item()
                val_num_batches += 1
            #print('precision = ', val_precision/val_num_batches)
            #print('recall = ', val_recall/val_num_batches)
            val_precisions.append(val_precision/val_num_batches)
            val_recalls.append(val_recall/val_num_batches)
            
    plt.plot(train_precisions, train_recalls)
    plt.plot(val_precisions, val_recalls)
    plt.title('Averaged Precision Recall for Molecular Function')
    plt.legend([f'train auc = {round(auc(train_precisions, train_recalls), 3)}', f'val auc = {round(auc(val_precisions, val_recalls), 3)}'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('test.png')
        
def main():
    
    logger = logging.getLogger('stdout')
    logger.setLevel(logging.DEBUG)
    hdlr = logging.StreamHandler(sys.stdout)
    logger.addHandler(hdlr)
    
    logger.debug('starting training')
    
    config_path = "../../../config/gcn.conf"
    conf = configparser.ConfigParser()
    conf.read(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_dir = conf['path']['gcn_data'] + '/data.pkl'
    if not os.path.exists(data_dir):
        data.main()
    try:
        with open(data_dir, 'rb') as f:
            data_dict = pickle.load(f)
            logger.debug('data successfully found')
    except:
        logger.error(f'can not find data at {data_dir}')
           
    vocab_path = conf['path']['vocab']
    with open(vocab_path, 'r') as of:
        aa_vocab = json.load(of)
        
    train_data, val_data, test_data, target_dim = split_data(data_dict, aa_vocab, 'mf') #change to variable GO type later
    logger.debug(f'size of training set = {len(train_data)}')
    logger.debug(f'size of validation set = {len(val_data)}')
    logger.debug(f'# of go terms = {target_dim}')
    
    if conf['path']['model'] is not None and conf['path']['model'] != '':
        
        model = torch.load(conf['path']['model'])
        
    else:
        
        lm_model_path = conf['path']['lm']
        language_model = torch.load(lm_model_path, map_location=device)
        lm_embedding = language_model['model_state_dict']['module.word_embeddings.weight']
        
        model = GCN(target_dim, lm_embedding)
        
    train(logger, model, train_data, val_data, target_dim)
    
    evaluate(model, train_data, val_data, target_dim)

if __name__ == '__main__':
    
    main()
    
        
    