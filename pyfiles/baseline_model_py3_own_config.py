#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, SimpleRNN, GRU, LSTM, Flatten, Embedding
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import random
import os
import csv
import argparse
from numpy import nan
matplotlib.use('Agg')




def get_model(max_seq_len, embedding_size, latent_dim, dense_size, lr, dropout):
    encoder_inputs = Input(shape=(max_seq_len, embedding_size), name='encoder_i')
    encoder = SimpleRNN(latent_dim, return_state=True, dropout=dropout, name='rnn1') #activation='relu',
    encoder_outputs, state_h = encoder(encoder_inputs, training=True)
    encoder_states = state_h
    decoder_inputs = Input(shape=(None, embedding_size), name='decoder_i')
    decoder_rnn = SimpleRNN(latent_dim, return_sequences=True, return_state=True, dropout=dropout, name='rnn2')           #activation='relu'
    decoder_outputs, _= decoder_rnn(decoder_inputs, initial_state=encoder_states, training=True)
    decoder_dense = Dense(dense_size, name='dense1') #activation='relu'
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='encoder-decoder')
    #model.compile(optimizer=Adam(learning_rate=lr), loss='mse') #run_eagerly=True
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return (model, (encoder, encoder_inputs, encoder_states, decoder_inputs, decoder_rnn, decoder_dense))



def inference_model(encoder, encoder_inputs, encoder_states, decoder_inputs, decoder_rnn, decoder_dense):
    encoder.training =  False
    encoder_model = Model(encoder_inputs, encoder_states, name='encoder')

    decoder_state_input_h = Input(shape=(latent_dim,))
    #decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = decoder_state_input_h #, decoder_state_input_c]
    decoder_outputs, state_h1 = decoder_rnn(
        decoder_inputs, initial_state=decoder_states_inputs, training=False)
    decoder_states = state_h1
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs , decoder_states_inputs],
        [decoder_outputs , decoder_states], name='decoder')
    return (encoder_model, decoder_model)


def decode_sequence(input_seq, input_length, encoder_model, decoder_model, max_seq_len, emd_dim, dense_size, max_op_seq_length):
    input_seq = input_seq.reshape(-1, max_seq_len, emd_dim)
    states_value = encoder_model.predict(input_seq)
    #print(states_value.shape)
    target_seq = input_length.reshape(-1, 1, dense_size) #(N * 1 * 1)
    #print(target_seq.shape)
    decoded_seq = []
    stop_condition = False
    count = 0
    while not stop_condition:
        output_tokens, h = decoder_model.predict(
            [target_seq, states_value])
        #print(output_tokens.shape)
        #print(h.shape)
        decoded_seq.append(output_tokens)
        count+=1
        #print(count)
        if count >= max_op_seq_length:
            stop_condition = True

        target_seq = output_tokens
        states_value = h
        #print(np.array(decoded_seq).shape)
    decoded_seq = np.hstack((decoded_seq))
    return decoded_seq.reshape(-1, max_op_seq_length, dense_size)


def train_model(train_dataset, model, epochs, batch_size):
    #train_dataset = np.log2(np.array([np.array(arr) for arr in train_dataset['output_seq_15'].values]))
    #print(train_dataset.shape)
    encoder_input_data  =  train_dataset[:, 0:11]
    decoder_input_data  =  train_dataset[:, 10:15]
    decoder_target_data =  train_dataset[:, 11:16]
    hist = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,             
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.3)
    return hist# model



def test_model(test_dataset, encoder_model, decoder_model, max_seq_len, emb_dim, dense_size, max_op_seq_length):
    #test_dataset = np.log2(np.array([np.array(arr) for arr in test_dataset['output_seq_15'].values]))
    test_data = test_dataset[:, 0:11]
    test_input_length = test_dataset[:, 10]
    test_actual_op = test_dataset[:, 11:16].reshape(-1,5,1)
    test_predicted_op = decode_sequence(test_data, test_input_length, encoder_model, decoder_model, max_seq_len, emb_dim, dense_size, max_op_seq_length)
    error = mse(test_predicted_op.reshape(-1,), test_actual_op.reshape(-1,))
    return error



def learning_curves(history, name):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    #plt.show()
    plt.savefig(name)
    return


def model_save(model, encoder, decoder, config):
    model.save('./exp_models/multi_baseline_nsc_'+config, save_format="h5") # nsc == not scaled output
    encoder.save('./exp_models/multi_baseline_enc_nsc_'+config, save_format="h5")# without nsc == output scaled
    decoder.save('./exp_models/multi_baseline_dec_nsc_'+config, save_format="h5")
    return


if __name__ == '__main__':
    print("Process started ...")
    #seed_value = 0
    #os.environ['PYTHONHASHSEED'] = str(seed_value)
    #random.seed(seed_value)
    #np.random.seed(seed_value)
    #tf.random.set_seed(seed_value)
    
    parser = argparse.ArgumentParser(description='Baseline Model')
    parser.add_argument('-sy','--start_date', help='train start year', required=True)
    parser.add_argument('-ey','--end_date', help='train end year', required=True)
    parser.add_argument('-tsy','--test_start_date', help='test start year', required=True)
    parser.add_argument('-tey','--test_end_date', help='test end year', required=True)
    args = vars(parser.parse_args())
    train_year = f'{args["start_date"]}_{args["end_date"]}'
    test_year = f'{args["test_start_date"]}_{args["test_end_date"]}'

    start_date = int(args['start_date'])  #1950
    end_date   = int(args['end_date'])    #1980
    test_start_date = int(args['test_start_date']) #1980
    test_end_date   = int(args['test_end_date'])   #1985
    
    print(f"Reading dataset...")
    dataset_loc = '../../dataset/basline_dataset2.csv'
    dataset = pd.read_csv(dataset_loc, sep=',', lineterminator="\n", low_memory=False)
    filtered_data = dataset[dataset['input_connected']==True].copy()
    print(f"filtered dataset shape : {filtered_data.shape}")
    del dataset
    filtered_data['input_nodes_sequence'] = filtered_data['input_node_years'].apply(lambda x : [i[0] if i else None for i in eval(x)])#new
    filtered_data['input_years_sequence'] = filtered_data['input_node_years'].apply(lambda x : [i[1] if i else None for i in eval(x)])#new
    filtered_data['first_year']= filtered_data['input_years_sequence'].apply(lambda x:x[1]) #new after 05-05-2022
    filtered_data['flag'] = filtered_data['first_year']+15 <= 2021 #new
    filtered_data = filtered_data[filtered_data['flag']==True].copy() #new
    
    latent_dim = 32 #64
    batch_size = 32
    lr = 0.0001 #0.0001
    epochs = 25 #10
    dropout = 0.02
    
    
    max_seq_len = 11
    embedding_size = 1
    dense_size = 1
    max_op_seq_length = 5
    
    save = True
    scaled = True
    
    print(f"latent dim : {latent_dim}, Batch size : {batch_size}, lr : {lr}, epochs : {epochs}, dropout : {dropout}, scaled : {scaled}")
    print(f"train year : {start_date}_{end_date}, test year : {test_start_date}_{test_end_date}")
    print(type(start_date))    
    training_data = filtered_data[(filtered_data['Year'] > start_date) & (filtered_data['Year'] <= end_date)].copy()
    testing_data = filtered_data[(filtered_data['Year'] > test_start_date) & (filtered_data['Year'] <= test_end_date)].copy()
    del filtered_data
    
    training_data['output_seq_15'] = training_data['output_seq_15'].apply(lambda x : eval(x))
    testing_data['output_seq_15']  = testing_data['output_seq_15'].apply(lambda x : eval(x))
    

    if scaled:
        train_dataset = np.log2(np.array([np.array(arr) for arr in training_data['output_seq_15'].values]))
        test_dataset  = np.log2(np.array([np.array(arr) for arr in testing_data['output_seq_15'].values]))
    else:
        train_dataset = np.array([np.array(arr) for arr in training_data['output_seq_15'].values])
        test_dataset  = np.array([np.array(arr) for arr in testing_data['output_seq_15'].values])
    # extra 1 added to make it similar to our apparoach but the citation logic not hold, so not added
    print(f"scaled data: {train_dataset[0]}")    
    print(f"train data shape : {train_dataset.shape}")
    print(f"test data shape : {test_dataset.shape}")
    
    
    model, other = get_model(max_seq_len, embedding_size, latent_dim, dense_size, lr, dropout)
    print("Model created")
    
    history = train_model(train_dataset, model, epochs, batch_size)
    print("Model trained")
    
    enc_dec = inference_model(*other)
    
    error = test_model(test_dataset, *enc_dec, max_seq_len, embedding_size, dense_size, max_op_seq_length)
    
    val_error = history.history['val_loss'][-1]
    
    config = f"{max_seq_len}_{embedding_size}_{batch_size}_{latent_dim}_{lr}_{epochs}_{train_year}_{test_year}_{dropout}_{scaled}_{val_error}_{error}"
    print(f"Test Error : {error}, Val Error : {val_error}")
    name = f"./exp_lc/baseline_own_32_{config}.png"
    learning_curves(history, name)
    print("learning curves created")
    
    with open('exp_results/basline_own_32_citation.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(config.split("_"))
    if save :
        #config = f"{batch_size}_{latent_dim}_{lr}_{epochs}_{train_year}_{test_year}_{val_error}_{error}_{scaled}"
        model_save(model, *enc_dec, config+"own_32")
        print(f"<<<<<= model saved =>>>>>")
    print("===================Finished===============")

