import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from model import GCN, Toy
import pickle
import configparser
import logging
import sys
import os
import json
import pdb_data
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np
import seaborn as sns

sys.path.append("../../../utils/")
from data import prepare_sequence

# global device param
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split_data(data_dict, aa_vocab, go_type = 'mf'):
    
    """
    if go_type == 'mf':
        X = data_dict['X_mf'] 
        y = data_dict['y_mf']  
        go_vocab = data_dict['mf_vocab']
        
    if go_type == 'bp':
        X = data_dict['X_bp'] 
        y = data_dict['y_bp']  
        go_vocab = data_dict['bp_vocab']
        
    if go_type == 'cc':
        X = data_dict['X_cc'] 
        y = data_dict['y_cc']  
        go_vocab = data_dict['cc_vocab']
    """
    for fam_type in ['mf', 'bp', 'cc']:
        X = data_dict[f'X_{fam_type}'] 
        y = data_dict[f'y_{fam_type}']  
        go_vocab = data_dict[f'{fam_type}_vocab']
        
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.125, random_state=41)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.125, random_state=41)
    
    dataset_train = prepare_dataset(X_train, y_train, go_vocab, aa_vocab)    
    dataset_val = prepare_dataset(X_val, y_val, go_vocab, aa_vocab)
    dataset_test = prepare_dataset(X_test, y_test, go_vocab, aa_vocab)    
        
    return dataset_train, dataset_val, dataset_test, len(go_vocab)

def prepare_dataset(X_data, y_data, go_vocab, aa_vocab):
    
    dataset = []
    
    for data, raw_labels in zip(X_data, y_data):
        x = prepare_sequence(data[0], aa_vocab, 3000).float()
        edge_index = from_scipy_sparse_matrix(data[1])[0]
        labels = torch.tensor([go_vocab[y] for y in raw_labels])
        targets = torch.zeros((len(go_vocab)))
        targets[labels] = 1
        targets = torch.tensor(targets)
        dataset.append(Data(x = x, edge_index = edge_index, y = targets))
        
    return dataset

def train(logger, model, dataset_train, dataset_val, target_dim, lr=0.00005, num_epoch = 100):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()
    train_loader = DataLoader(dataset_train, shuffle = True, batch_size=64)
    model.train()
    for epoch in range(num_epoch):
        for iteration, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.float().reshape(-1, target_dim))
            loss.backward()
            optimizer.step()
            #if iteration % 40 == 0:
            #    logger.debug(f'epoch {epoch + 1}, iteration {iteration}')
            #    logger.debug(f'loss = {loss}')
            #    pred = (out > 0.5).float()
            #    label = data.y.float().reshape(-1, target_dim)
            #    logger.debug(f'precision = {((label * pred).sum()/label.sum()).item()}')
            #    logger.debug(f'recall = {((label * pred).sum()/pred.sum()).item()}')
            #    logger.debug(f'total # of go terms = {label.sum().item()}')
            #    logger.debug(f'predicted # of go terms = {pred.sum().item()}')
            #    logger.debug('')
                
def evaluate(model, dataset, target_dim):
    
    model.eval()
    data_loader = DataLoader(dataset, batch_size=64)
    with torch.no_grad():
        
        logits = None
        labels = None
        
        for data in data_loader:
            data = data.to(device) 
            if logits is None:
                logits = model(data).float().cpu().numpy()
                labels = data.y.float().reshape(-1, target_dim).cpu().numpy()
            else:
                logits = np.vstack((logits, model(data).float().cpu().numpy()))
                labels = np.vstack((labels, data.y.float().reshape(-1, target_dim).cpu().numpy()))
            
    return logits, labels

def compute_pr(scores, labels, target_dim):
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(target_dim):
        precision[i], recall[i], _ = precision_recall_curve(labels[:, i], scores[:, i])
        average_precision[i] = average_precision_score(labels[:, i], scores[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(labels.ravel(), scores.ravel())
    average_precision["micro"] = round(average_precision_score(labels, scores, average="micro"), 3)
    
    return precision["micro"], recall["micro"], average_precision["micro"]
        
    
        
def main():
    
    logger = logging.getLogger('stdout')
    logger.setLevel(logging.DEBUG)
    hdlr = logging.StreamHandler(sys.stdout)
    logger.addHandler(hdlr)
    
    logger.debug('starting training')
    
    config_path = "../../../config/gcn.conf"
    conf = configparser.ConfigParser()
    conf.read(config_path)
    
    logger.debug(f'running on {device}')
    
    data_dir = conf['path']['gcn_data'] + '/data.pkl'
    if not os.path.exists(data_dir):
        pdb_data.main()
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
        
    plt.figure(figsize = (10, 6))
    
    lm_model_paths = conf['path']['lm']
    
    if type(lm_model_paths) != list:
        lm_model_paths = [lm_model_paths]
    
    for lm_model_path in lm_model_paths:
        
        language_model = torch.load(lm_model_path, map_location=device)
        lm_embedding = language_model['model_state_dict']['module.word_embeddings.weight']
                        
        num_trials = int(conf['model']['num_trials'])
        for i in range(num_trials):
            
            model_name = lm_model_path.split('/')[-1].split('.')[0] + '_GCN' + '_iter_{}'.format(i)
            
            if conf['model']['model'] == 'GCN':

                model = GCN(target_dim, lm_embedding)
            
            elif conf['model']['model'] == 'Toy':
                
                 model = Toy(target_dim, lm_embedding)
                

            model.to(device)
            
            logger.debug(f'traning on {model_name}')

            train(logger, model, train_data, val_data, target_dim)
            
            model_dir = conf['path']['models']
            torch.save(model, model_dir + '/' + model_name + '.model')

            scores, labels = evaluate(model, val_data, target_dim)
            
            precision, recall, ap_score = compute_pr(scores, labels, target_dim)
            
            logger.debug('Average precision score for {}: {}'.format(model_name, ap_score))
            
            plt.step(recall, precision, where='post', label = 'AP score for {}: {}'.format(model_name, ap_score))
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('micro-averaged PR curve over all classes')
    plt.legend()
    figure_dir = conf['path']['figures']
    plt.savefig(figure_dir + '/temp.png')   
    logger.debug('plot saved to: ' + figure_dir + '/temp.png')

if __name__ == '__main__':
    
    main()
    
        
    