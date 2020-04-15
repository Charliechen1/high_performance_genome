#!/usr/bin/env python
# coding: utf-8

import numpy as np
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
    for pdb_code in pdb_list.split(', '):
        logger.debug(f'Processing {pdb_code}')
        pdbl = Bio.PDB.PDBList()
        pdb_path = pdbl.retrieve_pdb_file(pdb_code, pdir = pdb_dir, file_format = 'pdb', overwrite = True)
        if os.path.exists(pdb_path):
            structure = Bio.PDB.PDBParser(QUIET = True).get_structure(pdb_code, pdb_path)
            model = structure[0]
            sequence = Bio.PDB.Selection.unfold_entities(model, 'R')
            if len(sequence) <= 3000:
                try:
                    ppb = Bio.PDB.CaPPBuilder()
                    for pp in ppb.build_peptides(structure):
                        sequence_dict[pdb_code] = str(pp.get_sequence())
                    dist_matrix = calc_dist_matrix(pp)
                    contact_map = np.array((dist_matrix < 8.0) & (dist_matrix > 0.01))*1
                    sparse_contact_map = scipy.sparse.coo_matrix(contact_map)
                    print(sparse_contact_map.shape)
                    scipy.sparse.save_npz(contact_map_dir + '/' + pdb_code + '.npz', sparse_contact_map)
                    num_successful_maps += 1
                    logger.debug(f'Successfully processed {pdb_code}')
                except:
                    e = sys.exc_info()[0]
                    logger.debug('Error for PDB code {}'.format(pdb_code), str(e))
                    num_failed_maps += 1
            else:
                logger.debug('Skipping PDB code {} because it is too long'.format(pdb_code))
                num_failed_maps += 1

            os.remove(pdb_path)
        else:
            logger.debug('{} not found'.format(pdb_code))
            
    json_data = json.dumps(sequence_dict)
    f = open(sequence_dir,"w")
    f.write(json_data)
    f.close()

    logger.debug('process finished')
    logger.debug('Number of maps generated: {}'.format(num_successful_maps))
    logger.debug('Number of pdb codes skipped: {}'.format(num_failed_maps))