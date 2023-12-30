#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
import argparse
matplotlib.use('Agg')
import numpy as np
import pickle
import csv


def get_model(max_seq_len, embedding_size, latent_dim, dense_size, lr, dropout):
    encoder_inputs = Input(shape=(max_seq_len, embedding_size))
    encoder = LSTM(latent_dim, return_state=True, dropout=dropout)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs, training=True)
    encoder_states = [state_h, state_c]


    decoder_inputs = Input(shape=(None, dense_size))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states, training=True)
    decoder_dense = Dense(dense_size)
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model, (encoder, encoder_inputs, encoder_states, decoder_lstm, decoder_inputs, decoder_dense)



def inference_model(encoder, encoder_inputs, encoder_states, decoder_lstm, decoder_inputs, decoder_dense):
    encoder.training = False
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs, training=False)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    return (encoder_model, decoder_model)


def decode_sequence(input_seq, input_length, encoder_model,decoder_model, max_seq_len, emb_dim, output_size, max_op_seq_length):
    input_seq = input_seq.reshape(-1, max_seq_len, emb_dim)
    states_value = encoder_model.predict(input_seq)

    target_seq = np.array(input_length).reshape(-1, 1, output_size) #(N * 1 * 1)
    #print(target_seq.shape)
    decoded_seq = []

    stop_condition = False
    count = 0
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        #print(output_tokens)
        decoded_seq.append(output_tokens)
        #print(output_tokens.shape)
        count+=1
        #print(count)
        if count >= max_op_seq_length:
            stop_condition = True

        target_seq = output_tokens
        states_value = [h, c]
    decoded_seq = np.hstack((decoded_seq))
    return decoded_seq.reshape(-1, max_op_seq_length, output_size)


def train_model(train_dataset, model, epochs, batch_size, scaled):
    if not scaled:
        mod_train_dataset = np.log2(train_dataset[2])
    else:
        mod_train_dataset = train_dataset[2]
    print(mod_train_dataset[0,:])

    encoder_input_data  =  train_dataset[0]
    decoder_input_data  =  mod_train_dataset[: ,0:5]
    decoder_target_data =  mod_train_dataset[: ,1:6]
    decoder_input_data  = decoder_input_data.reshape(-1,5,1)
    decoder_target_data = decoder_target_data.reshape(-1,5,1)#metric = ['accuracy'] run_eagerly=True
    hist = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.3,
              use_multiprocessing=True)
    return hist


def test_model(test_dataset, encoder_model, decoder_model, max_seq_len, emb_dim, output_size, max_op_seq_length,scaled):
    if not scaled:
        mod_test_dataset = np.log2(test_dataset[2])
    else:
        mod_test_dataset = test_dataset[2]
        
    test_data = test_dataset[0]
    test_input_length = mod_test_dataset[:,0]
    test_actual_op    = mod_test_dataset[:,1:].reshape(-1,5,1)
    test_predicted_op = decode_sequence(test_data, test_input_length, encoder_model, decoder_model, max_seq_len, emb_dim, output_size, max_op_seq_length)
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
    model_dir = './exp_models/'
    model.save(f"{model_dir}multi_op_{config}", save_format="h5")
    encoder.save(f"{model_dir}multi_op_enc_{config}", save_format="h5")
    decoder.save(f"{model_dir}multi_op_dec_{config}", save_format="h5")
    return


if __name__ == '__main__':
    print("Process started ...")
    
    parser = argparse.ArgumentParser(description='Method 2 Model')
    parser.add_argument('-sy','--start_date', help='train start year', required=True)
    parser.add_argument('-ey','--end_date', help='train end year', required=True)
    parser.add_argument('-tsy','--test_start_date', help='test start year', required=True)
    parser.add_argument('-tey','--test_end_date', help='test end year', required=True)
    args = vars(parser.parse_args())
    
    
    train_year = str(args['start_date'])+'_'+str(args['end_date'])
    test_year  = str(args['test_start_date'])+'_'+str(args['test_end_date'])

    train_name = f'./dataset/training_data_ns_gcn_{train_year}.pickle' #cheeck carefully before doing
    test_name  = f'./dataset/testing_data_ns_gcn_{test_year}.pickle'
    print(f"Reading dataset...")
    print(f"Train file name : {train_name}")
    print(f"Test file name : {test_name}")
    with open(train_name, 'rb') as f1:
        training_data = pickle.load(f1)
    with open(test_name, 'rb') as f2:
        testing_data = pickle.load(f2)
        
    print(f"Train data size : {training_data[0].shape}")
    print(f"Test data size : {testing_data[0].shape}")
    
       
    max_seq_len = training_data[0].shape[1]
    embedding_size = training_data[0].shape[2]
    latent_dim = 64
    batch_size = 32
    epochs = 25
    lr = 0.0001
    dropout = 0.02
    
    max_op_seq_length = 5
    dense_size = 1
    output_size = 1

    scaled = False
    save   = True
    print(training_data[2][0,:])

    config = f"{max_seq_len}_{embedding_size}_{batch_size}_{latent_dim}_{lr}_{epochs}_{train_year}_{test_year}_{dropout}_{scaled}"
    print(config)
    model, other = get_model(max_seq_len, embedding_size, latent_dim, dense_size, lr, dropout)
    print("Model created")
    
    history = train_model(training_data, model, epochs, batch_size,scaled)
    print("Model trained")
    append_text ="new_m2_32_gcn"
    
    name = f"./exp_lc/multi_op_{config}{append_text}.png"
    learning_curves(history, name)
    print("learning curves created")
    
    enc_dec = inference_model(*other)
    error = test_model(testing_data, *enc_dec, max_seq_len, embedding_size, output_size, max_op_seq_length, scaled)
    val_error = history.history['val_loss'][-1]
    print(f"Test Error : {error}, Validation Error : {val_error}")
    print("Result Writing ...")


    row = f"{config}_{val_error}_{error}"
    with open(f"./exp_results/method2{append_text}.csv", 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(row.split("_")) 
    if save :
        #config = f"{batch_size}_{latent_dim}_{lr}_{epochs}_{train_year}_{test_year}_{val_year}_{error}_{dropout}"
        model_save(model, *enc_dec, row+append_text)
        print(f"<<<<<= model saved =>>>>>")
    print("Finished")
