#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import import_ipynb
import pandas as pd
from util_func import *
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from dataprep.clean import clean_text, clean_country, clean_df
import argparse



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


def scibert_model(name='../codes/scibert_embed/scibert_scivocab_uncased'):
    model_version = name
    do_lower_case = True
    model = BertModel.from_pretrained(model_version)
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
    return model, tokenizer


def scibert_embed(text, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states.detach().numpy().mean(1)[0]


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='scibert embedding generation')
    parser.add_argument('-si','--start_index', help='start index', required=True)
    parser.add_argument('-ei','--stop_index', help='stop index', required=True)
    args = vars(parser.parse_args())
    cleaned_data     = clean_data()
    model, tokenizer = scibert_model()
    title_embed    = {}
    univ_embed     = {}
    country_embed  = {}
    msc_embed      = {}
    concat         = {}
    si = int(args['start_index'])
    ei = int(args['stop_index'])
    for index, row in tqdm(cleaned_data.iloc[si:ei,:].iterrows(), total=ei-si):
        thesis  = scibert_embed(row['title'], model, tokenizer)
        univ    = scibert_embed(row['university'], model, tokenizer)
        country = scibert_embed(row['country'], model, tokenizer)
        msc     = scibert_embed(row['msc'], model, tokenizer)
        title_embed[row["id"]]   = thesis
        univ_embed[row["id"]]    = univ
        country_embed[row["id"]] = country
        msc_embed[row["id"]]     = msc
        concat[row["id"]]        = np.concatenate((thesis, univ, country, msc))
        #print(concat[row["id"]].shape)
    save_obj(title_embed, f'title_only_scibert_complete_embedding_{si}_{ei}')
    save_obj(univ_embed, f'univ_only_scibert_complete_embedding_{si}_{ei}')
    save_obj(country_embed, f'country_only_scibert_complete_embedding_{si}_{ei}')
    save_obj(msc_embed, f'msc_only_scibert_complete_embedding_{si}_{ei}')
    save_obj(concat, f'concatenated_scibert_complete_embedding_{si}_{ei}')
    print('done')
