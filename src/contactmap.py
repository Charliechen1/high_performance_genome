#!/usr/bin/env python
# coding: utf-8

import numpy as np
import Bio.PDB
import os 
import scipy.sparse


def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    if Bio.PDB.is_aa(residue_one) and Bio.PDB.is_aa(residue_two):
        diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
        return np.sqrt(np.sum(diff_vector * diff_vector))
    else:
        return 0

def calc_dist_matrix(chain) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain), len(chain)), np.float)
    for row, residue_one in enumerate(chain) :
        for col, residue_two in enumerate(chain) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

if __name__ == '__main__': 
    
    config_path = "../config/contact_maps.conf"
    conf = configparser.ConfigParser()
    conf.read(config_path)

    filename = conf['path']['pdb_codes']
    pdb_dir = conf['path']['pdb_dir']
    contact_map_dir = conf['path']['contact_maps_dir']

    pdb_file = open(filename, 'r')
    pdb_list = pdb_file.read()
    for pdb_code in pdb_list.split(', '):
        try:
            pdbl = Bio.PDB.PDBList()
            pdb_path = pdbl.retrieve_pdb_file(pdb_code, pdir = pdb_dir, file_format = 'pdb', overwrite = True)
            structure = Bio.PDB.PDBParser(QUIET = True).get_structure(pdb_code, pdb_path)
            model = structure[0]
            sequence = Bio.PDB.Selection.unfold_entities(model, 'R')
            dist_matrix = calc_dist_matrix(sequence)
            contact_map = np.array((dist_matrix < 8.0) & (dist_matrix > 0.01))*1
            sparse_contact_map = scipy.sparse.coo_matrix(contact_map)
            scipy.sparse.save_npz(contact_map_dir + '/' + pdb_code + '.npz', sparse_contact_map)
        except:
            print('Failed for PDB code {}'.format(pdb_code)) #this should probably use a logger?



