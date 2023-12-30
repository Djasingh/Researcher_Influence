#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE
from util_func import load_obj, save_obj 


# In[31]:


def obtain_reduced_tsne_embedding():
    print("Process Started")
    embed_dict =  load_obj("combined_embed_0_268655")
    embed_array = np.array(list(embed_dict.values()))
    title_array = embed_array[:, 0:768]
    univ_array  = embed_array[:, 768:1536]
    country_array  = embed_array[:, 1536:2304]
    msc_array  = embed_array[:, 2304:3072]
    print("data read complete")
    array_list = [title_array, univ_array, country_array, msc_array]
    tsne = TSNE(n_components=100, learning_rate=200, init='random', method='exact', n_jobs=8)
    results = list(map(lambda x : tsne.fit_transform(x), array_list))
    reduced_array = np.hstack(results)
    print("embedding complete")
    reduced_dimension_dict =  dict(zip(embed_dict.keys(), reduced_array))
    save_obj(reduced_dimension_dict, "combined_reduced_tsne_embed")
    print("done")
    return reduced_dimension_dict


# In[32]:


if __name__=="__main__":
    reduced_dimension_dict = obtain_reduced_tsne_embedding()




