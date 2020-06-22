#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import Bio.PDB
import os 
import sys
import scipy.sparse
import configparser
import logging
import json
        
def calc_dist_matrix(sequence):
    X = []
    for res in sequence:
        X.append(res['CA'].coord)
    X = np.array(X)
    e = np.ones(X.shape[0]).reshape(-1, 1)
    s = np.sum(np.square(X), axis = 1).reshape(-1, 1)
    dist = np.sqrt((s @ e.T) + (e @ s.T) - (2 * X @ X.T))
    return dist


if __name__ == '__main__': 
    
    logger = logging.getLogger('stdout')
    logger.setLevel(logging.DEBUG)
    hdlr = logging.StreamHandler(sys.stdout)
    logger.addHandler(hdlr)
    
    config_path = "../../config/contact_maps.conf"
    conf = configparser.ConfigParser()
    conf.read(config_path)

    filename = conf['path']['pdb_codes']
    pdb_dir = conf['path']['pdb_dir']
    contact_map_dir = conf['path']['contact_maps_dir']
    sequence_dir = conf['path']['sequence_dir']
    
    logger.debug('process starting')
    
    if not os.path.exists(filename):
        
        snap_dir = conf['path']['snap_dir']
        df = pd.DataFrame()
        for file in os.listdir(snap_dir):
            df = pd.concat((df, pd.read_csv(snap_dir + '/' + file, 
                       sep = '\t', skiprows = 1, header = None,
                       names = ['NCBI taxid', 'Category', 'GO', 'String']
                       )))
        ensembl_file = conf['path']['ensembl']
        ensembl_to_pdb = pd.read_csv(ensembl_file, skiprows = 1)
        ensembl_to_pdb = ensembl_to_pdb[['PDB', 'TRANSLATION_ID']].drop_duplicates()
        df['TRANSLATION_ID'] = df['String'].astype(str).apply(lambda x : x.split('.')[1])
        df = df.merge(ensembl_to_pdb, on = 'TRANSLATION_ID')
        df['PDB'].drop_duplicates().to_csv(filename, sep = ',', header = False, index = False)
    
    if not os.path.exists(contact_map_dir):
        os.makedirs(contact_map_dir)
    pdb_file = open(filename, 'r')
    pdb_list = pdb_file.read()
    if os.path.exists(sequence_dir):
        with open(sequence_dir,"r") as f:
            sequence_dict = json.load(f)
            f.close()
    else:
        sequence_dict = {}
    num_successful_maps = 0
    num_failed_maps = 0
    
    start_index = 0
    
    for pdb_code in pdb_list.split('\n')[start_index:]:
        #logger.debug(f'Processing {pdb_code}')
        pdbl = Bio.PDB.PDBList()
        pdb_path = pdbl.retrieve_pdb_file(pdb_code, pdir = pdb_dir, file_format = 'pdb', overwrite = True)
        if os.path.exists(pdb_path) and not os.path.exists(contact_map_dir + '/' + pdb_code):
            structure = Bio.PDB.PDBParser(QUIET = True).get_structure(pdb_code, pdb_path)
        #    model = structure[0]
        #    sequence = Bio.PDB.Selection.unfold_entities(model, 'R')
        #    if len(sequence) <= 3000:
            try:
                ppb = Bio.PDB.CaPPBuilder()
                pp = ppb.build_peptides(structure)[0]
                sequence_dict[pdb_code] = str(pp.get_sequence())
                dist_matrix = calc_dist_matrix(pp)
                contact_map = np.array((dist_matrix < 8.0) & (dist_matrix > 0.01))*1
                sparse_contact_map = scipy.sparse.coo_matrix(contact_map)
                scipy.sparse.save_npz(contact_map_dir + '/' + pdb_code, sparse_contact_map)
                num_successful_maps += 1
                if num_successful_maps % 1000 == 0:
                    logger.debug(f'{num_successful_maps} proteins successfully processed')
                    json_data = json.dumps(sequence_dict)
                    f = open(sequence_dir,"w")
                    f.write(json_data)
                    f.close()
            except:
                e = sys.exc_info()[0]
                num_failed_maps += 1
        #    else:
        #        logger.debug('Skipping PDB code {} because it is too long'.format(pdb_code))
        #        num_failed_maps += 1

            os.remove(pdb_path)
        else:
            pass
    json_data = json.dumps(sequence_dict)
    f = open(sequence_dir,"w")
    f.write(json_data)
    f.close()

    logger.debug('process finished')
    logger.debug('Number of maps generated: {}'.format(num_successful_maps))
    logger.debug('Number of pdb codes skipped: {}'.format(num_failed_maps))