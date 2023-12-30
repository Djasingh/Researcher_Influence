#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
from dataprep.clean import clean_df
import numpy as np
from numpy import nan
from util_func import *
from tqdm import tqdm
import time
import argparse


def main(start_date, end_date, test_start_date, test_end_date):
    embeddings  = load_obj('combined_reduced_tsne_embed')

    print(f"embedding nodes : {len(embeddings)}")
    print(f"embedding shape : {embeddings[1].shape}")
    embeddings[0] = np.array([0]*embeddings[1].shape[0])
     
    data_loc = './inter_files/path_model_with_tree_older_rel_kept_input_output1.csv'
    data = pd.read_csv(data_loc, sep=',', lineterminator="\n", low_memory=False)
    inferred_dtypes, cleaned_data = clean_df(data)
    del data

    cleaned_data['first_year']= cleaned_data['input_years_sequence'].apply(lambda x:eval(x)[1]) #new after 05-05-2022
    cleaned_data['flag'] = cleaned_data['first_year']+15 <= 2021 #new
    cleaned_data = cleaned_data[cleaned_data['flag']==True].copy() #new
    cleaned_data['output_seq'] = cleaned_data['output_seq'].apply(lambda x : eval(x))
    cleaned_data['paths'] = cleaned_data['paths'].apply(lambda x : eval(x))

    num_paths = [len(paths) for paths in cleaned_data['paths'].values]
    max_num_paths = max(num_paths)
    print(f"Max no. paths : {max_num_paths}")
    path_length = [len(path) for paths in cleaned_data['paths'].values for path in paths]
    max_path_len = max(path_length)
    print(f"Max path len : {max_path_len}")
    padded_input_paths = []
    for index, row in tqdm(cleaned_data.iterrows(), total=cleaned_data.shape[0]):
        padded_path = pad_sequence(row['paths'], num_seq=max_num_paths, seq_length=max_path_len)
        padded_input_paths.append(padded_path) 
    cleaned_data['padded_paths'] = padded_input_paths
    
    train_data = cleaned_data[(cleaned_data['year'] > start_date) & (cleaned_data['year'] <= end_date)].copy()
    test_data = cleaned_data[(cleaned_data['year'] > test_start_date) & (cleaned_data['year'] <= test_end_date)].copy()
    del cleaned_data

    train_input = np.array([np.array(list1) for list1 in train_data['padded_paths'].values])
    train_output = np.array([np.array(list1) for list1 in train_data['output_seq'].values])
    test_input = np.array([np.array(list1) for list1 in test_data['padded_paths'].values])
    test_output = np.array([np.array(list1) for list1 in test_data['output_seq'].values])

    
    return embeddings, train_input, train_output, test_input, test_output, max_path_len, max_num_paths


def mapping(sequences):
    global embeddings
    global max_num_paths
    global max_path_len
    seq = np.array([embeddings[int(node)] for seq in sequences for node in seq])
    return seq.reshape(max_num_paths, max_path_len, -1)


if __name__ == "__main__":
    
    start = time.time()
    print("Process started...")
    parser = argparse.ArgumentParser(description='Method 3 input output generation')
    parser.add_argument('-sy','--start_date', help='train start year', required=True)
    parser.add_argument('-ey','--end_date', help='train end year', required=True)
    parser.add_argument('-tsy','--test_start_date', help='test start year', required=True)
    parser.add_argument('-tey','--test_end_date', help='test end year', required=True)
    args = vars(parser.parse_args())
    
    start_date = int(args['start_date'])
    end_date =   int(args['end_date'])
    test_start_date = int(args['test_start_date'])
    test_end_date = int(args['test_end_date'])
    
    embeddings, train_input, train_output, test_input, test_output, max_path_len, max_num_paths = main(start_date, end_date,
                                                                                                       test_start_date,
                                                                                                       test_end_date)
    train_input =  np.array(list(map(mapping, train_input)))
    print(f"train input shape: {train_input.shape}")
    train_ip_name = f"./dataset/train_input_m3_tsne_ns_tree_oldrk_{start_date}_{end_date}.npy"
    train_op_name  = f"./dataset/train_output_m3_tsne_ns_tree_oldnrk_{start_date}_{end_date}.npy" #nrk:new realtion kept in tree
    print(f"train ip saving in loc : {train_ip_name}, {train_input.shape}")
    print(f"train op saving in loc : {train_op_name}, {train_output.shape}")
    np.save(train_ip_name, train_input)
    np.save(train_op_name, train_output)
    test_input =  np.array(list(map(mapping, test_input)))
    print(f"test input shape: {test_input.shape}")
    
    test_ip_name = f"./dataset/test_input_m3_tsne_ns_tree_oldrk_{test_start_date}_{test_end_date}.npy"
    test_op_name  = f"./dataset/test_output_m3_tsne_ns_tree_oldrk_{test_start_date}_{test_end_date}.npy"
    print(f"test ip saving in loc : {test_ip_name}, {test_input.shape}")
    print(f"test op saving in loc : {test_op_name}, {test_output.shape}")
    np.save(test_ip_name, test_input)
    np.save(test_op_name, test_output)
    
    print("Process finished")
    stop = time.time()
    print('Time (in hr): ', (stop - start)/3600)
