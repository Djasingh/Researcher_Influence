#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten, Layer, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import load_model
from keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import matplotlib
import csv
import time
import argparse
matplotlib.use('Agg')

class AGG_LAYER(Layer):
    def __init__(self, num_seq, latent_dim, batch_size):
        super(AGG_LAYER, self).__init__()
        self.num_seq = num_seq
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        #w_init = tf.random_normal_initializer()
        #self.w = tf.Variable(
        #    initial_value=w_init(shape=(self.input_dim, self.units), dtype="float32"),
         #   trainable=True,
        #)
    def build(self, input_shape):
        #print(input_shape)
        self.w = self.add_weight("w",shape=[1, self.num_seq], trainable=True)#initializer="ones"
    
    def backend_reshape(self, x, shape=(-1, 47, 512)):
        return  tf.keras.backend.reshape(x, shape)

    def call(self, inputs):
        return self.backend_reshape(tf.matmul(self.w, self.backend_reshape(inputs,(-1, self.num_seq, self.latent_dim))), (-1, self.latent_dim))
    
    def get_config(self):
        return {"units": self.latent_dim}


def backend_reshape(x, max_seq_len, embedding_size):
    return  tf.keras.backend.reshape(x, shape=(-1, max_seq_len, embedding_size))


def get_model(batch_size, latent_dim, max_num_seq, max_seq_len, embedding_size, dense_size, lr, dropout):
    reshape = Lambda(backend_reshape, output_shape=(max_seq_len, embedding_size), arguments={'max_seq_len':max_seq_len,'embedding_size':embedding_size})
    encoder_inputs = Input(shape=(max_num_seq, max_seq_len, embedding_size))
    mod_en_in = reshape(encoder_inputs)
    encoder = SimpleRNN(latent_dim, return_state=True, dropout = dropout)
    encoder_outputs, state_h= encoder(mod_en_in, training = True)
    agg_layer_h = AGG_LAYER(num_seq=max_num_seq, latent_dim=latent_dim, batch_size=batch_size)
    #agg_layer_c = AGG_LAYER(num_seq=max_num_seq, latent_dim=latent_dim, batch_size=batch_size)
    state_h1 = agg_layer_h(state_h)
    #state_c1 = agg_layer_c(state_c)
    encoder_states = state_h1


    decoder_inputs = Input(shape=(None, dense_size))
    decoder_lstm = SimpleRNN(latent_dim, return_sequences=True, return_state=True, dropout = dropout)
    decoder_outputs, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states, training = True)
    decoder_dense = Dense(dense_size)
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse') 
    return model, (encoder, encoder_inputs, encoder_states, decoder_lstm, decoder_inputs, decoder_dense)


#Inference Code
def inference_model(encoder, encoder_inputs, encoder_states, decoder_lstm, decoder_inputs, decoder_dense):
    encoder.training = False
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    #decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = decoder_state_input_h
    decoder_outputs, state_h = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs, training=False)
    decoder_states = state_h
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs, decoder_states_inputs],
        [decoder_outputs, decoder_states])
    return (encoder_model, decoder_model)


def decode_sequence(encoder_model, decoder_model, input_seq, input_length, output_size):
    input_seq = input_seq.reshape(-1, max_num_seq, max_seq_len, embedding_size)
    states_value = encoder_model.predict(input_seq)

    target_seq = np.array(input_length).reshape(-1, 1, output_size) #(N * 1 * 1)
    #print(target_seq.shape)
    decoded_seq = []

    stop_condition = False
    count = 0
    while not stop_condition:
        output_tokens, h= decoder_model.predict(
            [target_seq, states_value])
        #print(output_tokens)
        decoded_seq.append(output_tokens)
        #print(output_tokens.shape)
        count+=1
        #print(count)
        if count >= max_seq_len:
            stop_condition = True

        target_seq = output_tokens
        states_value = h
    decoded_seq = np.hstack((decoded_seq))
    return decoded_seq.reshape(-1, max_seq_len, output_size)


def train_model(train_input, train_output, model, batch_size, epochs):
    encoder_input_data = train_input
    decoder_input_data = train_output[:, 0:5]
    decoder_target_data = train_output[:, 1:6]
    decoder_input_data = decoder_input_data.reshape(-1,5,1)
    decoder_target_data = decoder_target_data.reshape(-1,5,1) #metric = ['accuracy'] run_eagerly=True
    hist = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.3,
              use_multiprocessing=True)
    return hist, model

def test_model(test_input, test_output, encoder_model, decoder_model, output_size):
    test_data = test_input
    test_input_length = test_output[:,0]
    test_actual_op = test_output[:,1:].reshape(-1,5,1)
    test_predicted_op = decode_sequence(encoder_model, decoder_model, test_data, test_input_length, output_size)
    error = mse(test_predicted_op.reshape(-1,), test_actual_op.reshape(-1,))
    return error

def learning_curves(history, name):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    #plt.show()
    plt.savefig(name)
    return

def model_save(model, encoder, decoder, config1):
    model.save('./models/path_'+config1)
    encoder.save('./models/path_enc_'+config1)
    decoder.save('./models/path_dec_'+config1)
    return

if __name__ == '__main__':
    
    start = time.time()
    print("Process started with RNN...")
    parser = argparse.ArgumentParser(description='Method 3')
    parser.add_argument('-sy','--train_start_year', help='Train start year(1950)', required=True)
    parser.add_argument('-ey','--train_end_year', help='Train end year (1980)', required=True)
    parser.add_argument('-tsy','--test_start_year', help='Test start year (1980)', required=True)
    parser.add_argument('-tey','--test_end_year', help='Test end year (1985)', required=True)

    args = vars(parser.parse_args())
    #print(type(args["train_start_year"]))
    train_year = args['train_start_year']+"_"+args['train_end_year'] #"1950_1980"
    test_year  = args['test_start_year']+"_"+args['test_end_year']   #"1995_2000"
    scaled_op = True

    train_ip_name = f"../family_tree/train_input_{train_year}.npy"
    train_op_name = f"../family_tree/train_output_{train_year}.npy"
    print(f"train input name :{train_ip_name}")
    print(f"train output name :{train_op_name}")
    train_input = np.load(train_ip_name)
    print(f"train input size:{train_input.shape}")

    if scaled_op :
        train_output = np.log2(np.load(train_op_name))
    else:
        train_output = np.load(train_op_name)

    #train_output = train_output[0:100,...]
    print(f"train output size:{train_output.shape}")
    
    test_ip_name = f"../family_tree/test_input_{test_year}.npy"
    test_op_name = f"../family_tree/test_output_{test_year}.npy"
    print(f"test input name :{test_ip_name}")
    print(f"test output name :{test_op_name}")
    test_input = np.load(test_ip_name)
    
    if scaled_op:
        test_output = np.log2(np.load(test_op_name))
    else:
        test_output = np.load(test_op_name)

    print(f"test input size:{test_input.shape}")
    print(f"test output size:{test_output.shape}")
    #test_output = test_output[0:100,...]
    
    max_num_seq = train_input.shape[1]
    max_seq_len = train_input.shape[2]
    embedding_size = train_input.shape[3]

    latent_dim = 512
    batch_size = 32
    epochs = 25
    lr = 0.0001
    dropout = 0.02
    dense_size  = 1
    output_size = 1
    
    save = True

    print(f"latent dim : {latent_dim}, Batch size : {batch_size}, lr : {lr}, epochs : {epochs}, dropout : {dropout}, scaled : {scaled_op}")
    
    model, other = get_model(batch_size, latent_dim, max_num_seq, max_seq_len, embedding_size, dense_size, lr, dropout)
    print("Model created")
    history, model1 = train_model(train_input, train_output, model, batch_size, epochs)
    print("Model trained")
    enc_dec = inference_model(*other)
    
    error = test_model(test_input, test_output, *enc_dec, output_size)
    val_error = history.history['val_loss'][-1]
    
    print(f"Test Error : {error}, Val error : {val_error}")
    
    name = f"./lc/path_model_{batch_size}_{latent_dim}_{lr}_{epochs}_{train_year}_{test_year}_{error}_{scaled_op}_{dropout}.png"
    learning_curves(history, name)
    print(f"learning curves created in dir : {name}")

    if save :
        config1 = f"{batch_size}_{latent_dim}_{lr}_{epochs}_{train_year}_{test_year}_{val_error}_{error}_{scaled_op}_{dropout}"
        model_save(model1, *enc_dec, config1)
        print(f"model saved")
    with open('./results/method3_result.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(['tanh','RNN', max_num_seq, max_seq_len, embedding_size, batch_size, latent_dim, lr, epochs, train_year, test_year,
                         val_error, error, scaled_op, dropout])
    print("Process finished")
    stop = time.time()
    print('Time (in hr): ', (stop - start)/3600)




