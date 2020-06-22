import numpy as np
import torch
import scipy.sparse
import os
import pandas as pd
import numpy as np
import networkx
import obonet
import json
import pickle
import configparser
import logging
import sys
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def filter_go_terms_by_count(X, y, min_count = 25):
    
    counts = {}
    for labels in y:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
            
    removal_indices = []
    for i in range(len(y)):
        labels = y[i]
        for j in reversed(range(len(labels))):
            label = labels[j]
            if counts[label] < min_count:
                labels.remove(label)
                if len(labels) == 0:
                    removal_indices.append(i)

    for i in reversed(removal_indices):
        X.pop(i)
        y.pop(i)
        
    go_terms = set()
    for labels in y:
        for label in labels:
            go_terms.add(label)

    vocab = {mf: idx for idx, mf in enumerate(go_terms)}
    
    return X, y, vocab

def main():
    
    logger = logging.getLogger('stdout')
    logger.setLevel(logging.DEBUG)
    hdlr = logging.StreamHandler(sys.stdout)
    logger.addHandler(hdlr)
    
    logger.debug('starting data processing')
    
    config_path = "../../../config/gcn.conf"
    conf = configparser.ConfigParser()
    conf.read(config_path)
    
    # load preprocessed contact maps
    contact_maps = {}
    map_dir = conf['path']['contact_maps_dir']
    for file in os.listdir(map_dir):
        if file.endswith('.npz'):
            pdb_code = file.split('.')[0]
            contact_map = scipy.sparse.load_npz(map_dir + '/' + file)
            contact_map.resize((3000, 3000))
            contact_maps[pdb_code] = contact_map
            
    # load GO terms from Gene Ontology      
    url = conf['go']['obo_url']
    graph = obonet.read_obo(url)
    goid_to_category = {id_: data.get('namespace') for id_, data in graph.nodes(data=True)}
    
    # load PDB to GO conversion file from SIFTS
    csv_file = conf['path']['csv_file']
    df = pd.read_csv(csv_file, skiprows = 1, error_bad_lines=False, warn_bad_lines = False)
    df = df[['PDB', 'GO_ID']]
    pdb_to_go = {}
    for key, value in df.values:
        go_list = pdb_to_go.get(key, set())
        go_list.add(value)
        pdb_to_go[key] = go_list
        
    # load preprocessed PDB sequences
    sequence_json = conf['path']['sequence_dir']
    with open(sequence_json, 'r') as f:
        sequence_dict = json.load(f)
    
    # find all GO terms per PDB ID
    X_mf_data = []
    y_mf_data = []
    X_bp_data = []
    y_bp_data = []
    X_cc_data = []
    y_cc_data = []
    for key, value in contact_maps.items():
        mf_go_list = []
        bp_go_list = []
        cc_go_list = []
        for go_term in pdb_to_go.get(key, []):
            if goid_to_category.get(go_term, '') == 'molecular_function':
                mf_go_list.append(go_term)
            if goid_to_category.get(go_term, '') == 'biological_process':
                bp_go_list.append(go_term)
            if goid_to_category.get(go_term, '') == 'cellular_component':
                cc_go_list.append(go_term)
        if len(mf_go_list) > 0:
            X_mf_data.append((sequence_dict[key], value))
            y_mf_data.append(mf_go_list)
        if len(bp_go_list) > 0:
            X_bp_data.append((sequence_dict[key], value))
            y_bp_data.append(bp_go_list)
        if len(cc_go_list) > 0:
            X_cc_data.append((sequence_dict[key], value))
            y_cc_data.append(cc_go_list)
            
    
    # remove GO terms with less than 25 representatives
    min_count = 25
    logger.debug(f'filtering out terms with less than {min_count} representatives')

    X_mf_data, y_mf_data, mf_vocab = filter_go_terms_by_count(X_mf_data, y_mf_data, min_count)
    logger.debug(f'# of mf terms = {len(mf_vocab)}, # of proteins = {len(X_mf_data)}')

    X_bp_data, y_bp_data, bp_vocab = filter_go_terms_by_count(X_bp_data, y_bp_data, min_count)
    logger.debug(f'# of bp terms = {len(bp_vocab)}, # of proteins = {len(X_bp_data)}')

    X_cc_data, y_cc_data, cc_vocab = filter_go_terms_by_count(X_cc_data, y_cc_data, min_count)
    logger.debug(f'# of cc terms = {len(cc_vocab)}, # of proteins = {len(X_cc_data)}')
    
    # save data
    data = {}
    data['X_mf'] = X_mf_data
    data['y_mf'] = y_mf_data
    data['mf_vocab'] = mf_vocab
    data['X_bp'] = X_mf_data
    data['y_bp'] = y_mf_data
    data['bp_vocab'] = mf_vocab
    data['X_bp'] = X_mf_data
    data['y_bp'] = y_mf_data
    data['bp_vocab'] = mf_vocab
    
    data_dir = conf['path']['gcn_data']
    if not os.path.exists(data_dir):
        os.makedir(data_dir)
    with open(data_dir + '/data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    logger.debug(f'finished data processing. data saved to {data_dir}/data.pkl')
    
if __name__ == '__main__':
    
    main()
