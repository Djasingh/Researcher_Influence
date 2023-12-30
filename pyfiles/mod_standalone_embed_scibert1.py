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
tqdm.pandas()


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

def mod_scibert_embed(text, model, tokenizer):
    text_embed = []
    for i in text[['title','university','country','msc']].values:
        input_ids = torch.tensor(tokenizer.encode(i)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]# The last hidden-state is the first element of the output tuple
        text_embed.append(last_hidden_states.detach().numpy().mean(1)[0])
    return np.array(text_embed).T


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
    si = int(args['start_index'])
    ei = int(args['stop_index'])

    cleaned_data     = clean_data()
    model, tokenizer = scibert_model()
    cleaned_data.loc[si:ei,'thesis_embed'] = cleaned_data.loc[si:ei, 'title'].progress_apply(scibert_embed, model=model, tokenizer=tokenizer) 
    cleaned_data.loc[si:ei,'univ_embed'] = cleaned_data.loc[si:ei, 'university'].progress_apply(scibert_embed, model=model, tokenizer=tokenizer) 
    cleaned_data.loc[si:ei,'country_embed'] = cleaned_data.loc[si:ei, 'country'].progress_apply(scibert_embed, model=model, tokenizer=tokenizer) 
    cleaned_data.loc[si:ei,'msc_embed'] = cleaned_data.loc[si:ei, 'msc'].progress_apply(scibert_embed, model=model, tokenizer=tokenizer)
    embed_data = cleaned_data.loc[si:ei, ['id','thesis_embed', 'univ_embed','country_embed','msc_embed']].to_numpy()
    filename =  f"standalone_embed_{si}_{ei}"
    np.save(filename, embed_data)
    print('done')
