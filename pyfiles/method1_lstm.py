#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model
from mlflow import log_metric, log_param, log_artifacts
import matplotlib.pyplot as plt
import numpy as np
import pickle
import mlflow
import argparse
import csv

def get_model(seq_len, emb_dim, lr, hidden_units, dense_size, dropout):
    i = Input(shape=(seq_len, emb_dim))
    lstm = LSTM(hidden_units, dropout=dropout, return_state = True, name='lstm')
    _,h,c = lstm(i, training=True)
    #dense1 = Dense(128, activation='relu', name='dense1')
    #dense2 = Dense(64, name='dense2')
    x = Concatenate(axis=1)([h, c])
    dense3 = Dense(dense_size, name='dense3')
    #x = dense1(x)
    #x = dense2(x)
    x = dense3(x)
    model = Model(i, x)
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=lr)
    )
    return (model, lstm)

def model_eval(test_dataset, model, lstm, batch_size):
    lstm.training = False
    predicted_op = model.predict(test_dataset[0], batch_size = batch_size).reshape(-1,)
    actual_op = np.log2(test_dataset[1])
    error = mse(predicted_op, actual_op)
    return error

def model_train_test(train_dataset, test_dataset, seq_len, emb_dim, lr, batch_size, epochs, hidden_units, dense_size, dropout):
    train_output =  np.log2(train_dataset[1])
    df_train, df_test, Ytrain, Ytest = train_test_split(train_dataset[0], train_output, test_size=0.30)
    model_param = get_model(seq_len, emb_dim, lr, hidden_units, dense_size, dropout)
    
    with mlflow.start_run():
        hist = model_param[0].fit(
            df_train, Ytrain,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(df_test, Ytest),
            use_multiprocessing=True
        )
        test_error = model_eval(test_dataset, *model_param, batch_size)
        log_param("epochs", epochs)
        log_param("train_size", train_dataset[0].shape)
        log_param("test_size", test_dataset[0].shape)
        log_param("emb_size", emb_dim)
        log_param("batch_size", batch_size )
        log_param("hidden_units", hidden_units)
        log_param("lr", lr)
        log_param("dropout", dropout)
        val_error = hist.history['val_loss'][-1]
        log_metric("val_mse", val_error)
        log_metric("test_mse", test_error)
    return (model_param[0], hist, test_error, val_error)

def learning_curves(history, name):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(name)
    return

def model_save(model, config):
    model.save('./exp_models/new_32_method1_'+config, save_format="h5") # nsc == not scaled output
    return



if __name__ == '__main__':
    print(f"Started ResIP-M1...")
    parser = argparse.ArgumentParser(description='Method 1')
    parser.add_argument('-sy','--start_date', help='train start year', required=True)
    parser.add_argument('-ey','--end_date', help='train end year', required=True)
    parser.add_argument('-tsy','--test_start_date', help='test start year', required=True)
    parser.add_argument('-tey','--test_end_date', help='test end year', required=True)
    args = vars(parser.parse_args())
    train_year = str(args['start_date'])+'_'+str(args['end_date'])
    test_year  = str(args['test_start_date'])+'_'+str(args['test_end_date'])
    
    train_name = f'./dataset/training_data_ns_{train_year}.pickle'
    test_name  = f'./dataset/testing_data_ns_{test_year}.pickle'
    with open(train_name, 'rb') as f1:
        training_data = pickle.load(f1)
    with open(test_name, 'rb') as f2:
        testing_data = pickle.load(f2)
    
    seq_len = training_data[0].shape[1]
    emb_dim = training_data[0].shape[2]
    
    hidden_units = 32
    batch_size = 64
    lr = 0.0001
    epochs = 25
    dropout = 0.02
    
    dense_size = 1
    
    save = True
    
    config = f"{seq_len}_{batch_size}_{hidden_units}_{lr}_{epochs}_{dropout}_{train_year}_{test_year}"
    print(config)
    print(f"Training testing started...")
    model, hist, test_error, val_error = model_train_test(training_data, testing_data, seq_len, emb_dim, lr, batch_size, epochs, hidden_units, dense_size, dropout)
    print(f"test_error:{test_error}, val_error : {val_error}")
    name = f"./exp_lc/new_32_method1_{config}.png"
    print(f"creating learning curve : {name}")
    learning_curves(hist, name)
    row = f"{config}_{val_error}_{test_error}"
    with open('./exp_results/new_32_method1.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(row.split("_"))
    if save :
        model_save(model, row)
        print(f"<<<<<= model saved =>>>>>")
    print(f"Finsihed")
