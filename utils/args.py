import numpy as np
import torch
import logging
import sys

def _parse_cuda_index_helper(s):
    try:
        i = int(s)
        if i > torch.cuda.device_count() or i < 0:
            raise ValueError(s)
        return i
    except :
        devices = str(np.arange(torch.cuda.device_count()))
        raise argparse.ArgumentTypeError(f'{s} is not a valid CUDA index. Please choose from {devices}')
        
        
def parse_cuda_index(string):
    if string == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        if ',' in string:
            return [_parse_cuda_index_helper(_) for _ in string.split(',')]
        else:
            return _parse_cuda_index_helper(string)
        

def parse_logger(string):
    if not string:
        ret = logging.getLogger('stdout')
        hdlr = logging.StreamHandler(sys.stdout)
    else:
        ret = logging.getLogger(string)
        hdlr = logging.FileHandler(string)
    ret.setLevel(logging.INFO)
    ret.addHandler(hdlr)
    hdlr.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    return ret