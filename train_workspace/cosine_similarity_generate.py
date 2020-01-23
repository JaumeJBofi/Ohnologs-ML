from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import pickle as pkl
import random
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


working_level = "Intermediate"
dataList = ["Ohnologs","No-Ohnologs","Paralogs"]
dataNameList = ["ohnologs","no-ohnologs","paralog"]

def get_embeddings(kmer,kind):    
    df_emb = {}
    if(kind != ""):
        kind = "_" + kind   
    for data in dataList:
        df_emb[data] = []
        df_emb[data].append(pd.read_pickle(working_level + "/embeddings_dna2vec/" + str(kmer) + "kmer/" + data.lower() + "1-vec-"+ str(kmer) + "kmer" + kind + ".pkl"))
        df_emb[data].append(pd.read_pickle(working_level + "/embeddings_dna2vec/" + str(kmer) + "kmer/" + data.lower() + "2-vec-"+ str(kmer) + "kmer" + kind + ".pkl"))
    return df_emb

from sklearn.metrics.pairwise import cosine_similarity
kmers_List = [3,8]
emb_type_List = ["","complete","cdna","cdna2"]
for curr_kmer in kmers_List:
    for cur_emb_type in emb_type_List:      
        emb_dict = get_embeddings(curr_kmer,cur_emb_type)
        for dataName in dataList:
            curr_df = emb_dict[dataName]
            cosine_similarity_list = []
            for i in range(0,len(curr_df[0].values)):    
                cosine_similarity_list.append(cosine_similarity([curr_df[0].values[i]], [curr_df[1].values[i]])[0][0])            
            df_save = pd.DataFrame()
            df_save["Cosine_Similarity"] = cosine_similarity_list      
            if(cur_emb_type == ""):
                cur_emb_type = "normal"
            df_save.to_pickle(working_level + "/cosine_similarity/" + dataName + "-" + cur_emb_type + "-" + str(curr_kmer) + ".pkl")                            