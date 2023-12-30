#!/usr/bin/env python
# coding: utf-8

import copy
import pandas as pd
import import_ipynb
import networkx as nx
import itertools
import numpy as np
from scipy import sparse
from tqdm import tqdm
from node2vec import Node2Vec
from dataprep.clean import clean_df
import collections
from util_func import save_obj
#%config Completer.use_jedi = False


def mod_get_paths(edge_list, root):
    '''
    return list of shortest paths from root node to every nodes (except root)/list of list(path)
    '''
    G1 = nx.DiGraph()
    G1.add_edges_from(edge_list)
    all_node = [node for node in G1 if node != root]
    paths = [nx.shortest_path(G1, root, node) for node in all_node]
    return paths


def mod_get_all_paths(edge_list, root):
    '''
    return list of all paths from root node to every nodes (except root)/list of list(path)
    '''
    G1 = nx.DiGraph()
    G1.add_edges_from(edge_list)
    all_node = [node for node in G1 if node != root]
    all_paths = [nx.all_simple_paths(G1, root, node) for node in all_node]
    paths = [p for paths in all_paths for p in paths]
    return paths


# In[16]:


#edges= [(0, 1), (1, 2), (0, 3), (3, 2),(0,4)]


# In[18]:


#mod_get_all_paths([(1,2),(1,3),(2,4),(2,5),(2,6),(3,6),(3,7)], 1)


# In[3]:


def cleaned_data():
    data_loc = './inter_files/input-output-data1_old_kept.csv'
    data = pd.read_csv(data_loc, sep=',', lineterminator="\n", low_memory=False)
    inferred_dtypes, cleaned_data = clean_df(data)
    return cleaned_data

def obtain_paths(cleaned_data):
    paths = {}
    input_paths = []
    output_seq  = []
    for index, row in tqdm(cleaned_data.iterrows(), total=cleaned_data.shape[0]):
        edge = eval(row["tree_edges"]) #check it (it should be tree_edges: 10-10-2022)
        idd = row['nodeid']
        path = mod_get_all_paths(edge, idd) # return list_of_lists
        paths[idd] = path
        input_paths.append(path)
        output_seq.append(eval(row['output_seq']))
        del path, edge
    cleaned_data['paths'] = input_paths
    save_obj(paths, 'family_paths')
    cleaned_data.to_csv('./inter_files/path_model_with_tree_older_rel_kept_input_output1.csv', index = False)
    return paths



def get_parameters():
    num_paths = [len(paths[idd]) for idd in paths] # max no. of paths
    path_length = [len(path) for idd in paths for path in paths[idd]] #max. path length
    num_paths_dist = collections.Counter(num_paths)
    max_num_paths = max(num_paths)
    path_length_dist = collections.Counter(path_length)
    max_path_len = max(path_length)
    return max_num_paths, max_path_len


if __name__=='__main__':
    clean_data = cleaned_data()
    paths = obtain_paths(clean_data)
    max_num_paths, max_path_len = get_parameters()
    print(f"max_num_paths : {max_num_paths}, max_path_len : {max_path_len}")