import random
import json
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import torch
import collections
import os


def get_base_dir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(root_dir)
    # root_dir = '~/PycharmProjects/doduo-privacy'
    return root_dir



def get_valid_types(data_name):

    with open('./data/types.json', 'r') as f:
        valid_types = json.load(f)

    if data_name in ['msato0', 'msato1', 'msato2', 'msato3', 'msato4', 'sato0', 'sato1', 'sato2', 'sato3', 'sato4']:
        return valid_types['type78']
    elif data_name =='sota-schema' or data_name == 'sota-schema-small' or data_name == 'dataset-sota' or data_name == 'dataset-sota-small':
        return valid_types['sota-schema']
    elif data_name =='sota-dbpedia' or data_name =='dataset-sota-dbpedia':
        return valid_types['sota-dbpedia']
    elif data_name == 'dataset-t2d':
        return valid_types['t2d']
    elif data_name == 'dataset-limaye':
        return valid_types['limaye']
    elif data_name == 'dataset-wikipedia':
        return valid_types['wikipedia']
    return valid_types[data_name]


def parse_tagname(tag_name):
    """sato_bert_bert-base-uncased-bs16-ml-256"""
    if "__" in tag_name:
        # Removetraining ratio
        tag_name = tag_name.split("__")[0]
    tokens = tag_name.split("_")[-1].split("-")
    shortcut_name = "-".join(tokens[:-3])
    max_length = int(tokens[-1])
    batch_size = int(tokens[-3].replace("bs", ""))
    return shortcut_name, batch_size, max_length


def set_seed(seed: int):
    """https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L58-L63"""    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """Add the following 2 lines
    https://discuss.pytorch.org/t/how-could-i-fix-the-random-seed-absolutely/45515
    """
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    """
    For detailed discussion on the reproduciability on multiple GPU
    https://discuss.pytorch.org/t/reproducibility-over-multigpus-is-impossible-until-randomness-of-threads-is-controled-and-yet/47079
    """
