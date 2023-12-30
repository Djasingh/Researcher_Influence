#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
#import import_ipynb
import pickle
import matplotlib.pyplot as plt
from util_func import load_obj
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse
from numpy import nan
#from dataprep.eda import create_report, plot, plot_missing



def load_embed():
    embedding = load_obj('family_gcn_embedding')# mod_embedding_family_graph_node2vec, node2vec_scibert_embed
    zere_dict= {'0': np.array([0]*embedding[1]['31'].shape[0])}
    embedding[0] = zere_dict
    return embedding


def input_read():
    data_loc = './inter_files/input-output-data1.csv'
    data = pd.read_csv(data_loc, sep=',', lineterminator="\n", low_memory=False)
    data['input_nodes_sequence'] = data['input_nodes_sequence'].apply(lambda x : eval(x))
    data['output_seq'] = data['output_seq'].apply(lambda x : eval(x))
    data['output'] = data['output_seq'].apply(lambda x : x[-1])
    data['first_year']= data['input_years_sequence'].apply(lambda x:eval(x)[1]) #new
    data['flag'] = data['first_year']+15 <= 2021 #new
    data = data[data['flag']==True].copy() #new
    node_seq_len = [len(a) for a in data['input_nodes_sequence'].values]
    max_len = max(node_seq_len)
    return data, max_len


def train_test_data(data, sy, ey, tsy, tey):
    start_date = sy
    end_date = ey
    test_start_date = tsy
    test_end_date = tey
    train_data = data[(data['Year'] > start_date) & (data['Year'] <= end_date)].copy()
    test_data = data[(data['Year'] > test_start_date) & (data['Year'] <= test_end_date)].copy()
    return train_data, test_data


def input_data(embedding, data_df, max_len):
    data = data_df[['nodeid','input_nodes_sequence']].values
    padded_data = pad_sequences(data[:,1], padding='post', maxlen=max_len)
    padded_data = [[i[0],x] for i,x in zip(data, padded_data)]
    data_emb = [embedding[list1[0]][str(node)] if node != 0 else embedding[0]['0'] for list1 in padded_data for node in list1[-1]]
    data_emb = np.array(data_emb)
    data_emb = data_emb.reshape(-1, max_len,embedding[1]['31'].shape[0])
    single_output_data = data_df['output'].values
    multi_output_data  = np.array([np.array(a) for a in data_df['output_seq'].values])
    multi_output_data  = multi_output_data
    
    return (data_emb, single_output_data, multi_output_data)


if __name__ == "__main__":
    print("started")
    parser = argparse.ArgumentParser(description='input output for m1 and m2')
    parser.add_argument('-sy','--start_date', help='train start year',required=True)
    parser.add_argument('-ey','--end_date', help='train end year', required=True)
    parser.add_argument('-tsy','--test_start_date', help='test start year', required=True)
    parser.add_argument('-tey','--test_end_date', help='test end year', required=True)
    args = vars(parser.parse_args())
    start_date = int(args['start_date'])
    end_date   = int(args['end_date'])
    test_start_date = int(args['test_start_date'])
    test_end_date   = int(args['test_end_date'])
    embedding = load_embed()
    data, max_len = input_read()
    train_data, test_data= train_test_data(data, start_date, end_date, test_start_date, test_end_date)
    training_data = input_data(embedding, train_data, max_len)
    testing_data  = input_data(embedding, test_data, max_len)
    print(f"training info : {start_date}_{end_date}, {training_data[0].shape}")
    print(f"testing data size :{test_start_date}_{test_end_date}, {testing_data[0].shape}")
    train_name    = f"./dataset/training_data_ns_gcn_{start_date}_{end_date}.pickle"
    test_name     = f"./dataset/testing_data_ns_gcn_{test_start_date}_{test_end_date}.pickle"
    with open(train_name, 'wb') as f:
        pickle.dump(training_data, f, protocol=4)
    with open(test_name, 'wb') as f:
        pickle.dump(testing_data, f, protocol=4)
    print("finished...")
