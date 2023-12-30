#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import pandas as pd
from util_func import *
from tqdm import tqdm
#from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from dataprep.clean import clean_text, clean_country, clean_df
import argparse
#from tensorflow.keras.preprocessing.sequence import pad_sequences
tqdm.pandas()
cores = 8
torch.set_num_threads(cores)

def clean_data():
    data_loc ="../codes/new_data/complete_data-input-output.csv"
    data = pd.read_csv(data_loc, sep=',', lineterminator="\n", low_memory=False)
    inferred_dtypes, cleaned_data = clean_df(data)
    cleaned_data = clean_text(cleaned_data, "title")
    cleaned_data = clean_text(cleaned_data, "msc")
    cleaned_data = clean_text(cleaned_data, "university")
    cleaned_data = clean_text(cleaned_data, "country")

#     cleaned_data["title_c_u_msc"] = cleaned_data['title'].apply(str)+" "+cleaned_data['country'].apply(str)+" "+
#     cleaned_data['university'].apply(str)+" "+cleaned_data['msc'].apply(str)
    return cleaned_data

def mod_scibert_embed(row, model, tokenizer):
    results = [scibert_embed(text, model, tokenizer) for text in row.values]
    return np.array(results).flatten()


def scibert_model(name='../codes/scibert_embed/scibert_scivocab_uncased'):
    model_version = name
    do_lower_case = True
    model = BertModel.from_pretrained(model_version)
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
    return model, tokenizer

def scibert_model_update(name='allenai/scibert_scivocab_uncased'):
    model_version = name
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    model = AutoModel.from_pretrained(model_version)
    return model, tokenizer


def scibert_embed(row, model, tokenizer):
    input_ids = tokenizer(list(row.values), return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    #print(tokens)
    #input_ids = torch.tensor(pad_sequences(tokens['input_ids'])) # Batch size 4
    outputs = model(**input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states.detach().numpy().mean(1).flatten()


if __name__=='__main__':
    print(f"No. of cores using :{cores}")
    parser = argparse.ArgumentParser(description='scibert embedding generation')
    parser.add_argument('-si','--start_index', help='start index', required=True)
    parser.add_argument('-ei','--stop_index', help='stop index', required=True)
    args = vars(parser.parse_args())
    si = int(args['start_index'])
    ei = int(args['stop_index'])

    cleaned_data     = clean_data()
    model, tokenizer = scibert_model_update()
    cleaned_data.loc[si:ei,'combined_embed'] = cleaned_data.loc[si:ei, ['title','university','country','msc']].progress_apply(scibert_embed, model=model, tokenizer=tokenizer, axis=1) 
    embed_data = cleaned_data.loc[si:ei, ['id','combined_embed']].set_index('id')['combined_embed'].to_dict()
    filename =  f"combined_embed_{si}_{ei}"
    save_obj(embed_data, filename)
    print('done')
