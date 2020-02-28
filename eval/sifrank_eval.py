#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/25

import nltk
from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus
from util import fileIO
from stanfordcorenlp import StanfordCoreNLP
import time

def get_PRF(num_c, num_e, num_s):
    F1 = 0.0
    P = float(num_c) / float(num_e)
    R = float(num_c) / float(num_s)
    if (P + R == 0.0):
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)
    return P, R, F1


def print_PRF(P, R, F1, N):

    print("\nN=" + str(N), end="\n")
    print("P=" + str(P), end="\n")
    print("R=" + str(R), end="\n")
    print("F1=" + str(F1))
    return 0


time_start = time.time()

P = R = F1 = 0.0
num_c_5 = num_c_10 = num_c_15 = 0
num_e_5 = num_e_10 = num_e_15 = 0
num_s = 0
lamda = 0.0

database1 = "Inspec"
database2 = "Duc2001"
database3 = "Semeval2017"

database = database1

if(database == "Inspec"):
    data, labels = fileIO.get_inspec_data()
    lamda = 0.6
    elmo_layers_weight = [0.0, 1.0, 0.0]
elif(database == "Duc2001"):
    data, labels = fileIO.get_duc2001_data()
    lamda = 1.0
    elmo_layers_weight = [1.0, 0.0, 0.0]
else:
    data, labels = fileIO.get_semeval2017_data()
    lamda = 0.6
    elmo_layers_weight = [1.0, 0.0, 0.0]

#download from https://allennlp.org/elmo
options_file = "../auxiliary_data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "../auxiliary_data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

porter = nltk.PorterStemmer()#please download nltk
ELMO = word_emb_elmo.WordEmbeddings(options_file, weight_file, cuda_device=0)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=lamda, database=database)
en_model = StanfordCoreNLP(r'E:\Python_Files\stanford-corenlp-full-2018-02-27',quiet=True)#download from https://stanfordnlp.github.io/CoreNLP/

try:
    for key, data in data.items():

        lables = labels[key]
        lables_stemed = []

        for lable in lables:
            tokens = lable.split()
            lables_stemed.append(' '.join(porter.stem(t) for t in tokens))

        print(key)

        dist_sorted = SIFRank(data, SIF, en_model, elmo_layers_weight=elmo_layers_weight,if_DS=True,if_EA=True)
        # dist_sorted = SIFRank_plus(data, SIF, en_model, elmo_layers_weight=elmo_layers_weight)

        j = 0
        for temp in dist_sorted[0:15]:
            tokens = temp[0].split()
            tt = ' '.join(porter.stem(t) for t in tokens)
            if (tt in lables_stemed or temp[0] in labels[key]):
                if (j < 5):
                    num_c_5 += 1
                    num_c_10 += 1
                    num_c_15 += 1

                elif (j < 10 and j >= 5):
                    num_c_10 += 1
                    num_c_15 += 1

                elif (j < 15 and j >= 10):
                    num_c_15 += 1
            j += 1

        if (len(dist_sorted[0:5]) == 5):
            num_e_5 += 5
        else:
            num_e_5 += len(dist_sorted[0:5])

        if (len(dist_sorted[0:10]) == 10):
            num_e_10 += 10
        else:
            num_e_10 += len(dist_sorted[0:10])

        if (len(dist_sorted[0:15]) == 15):
            num_e_15 += 15
        else:
            num_e_15 += len(dist_sorted[0:15])

        num_s += len(labels[key])

    en_model.close()
    p, r, f = get_PRF(num_c_5, num_e_5, num_s)
    print_PRF(p, r, f, 5)
    p, r, f = get_PRF(num_c_10, num_e_10, num_s)
    print_PRF(p, r, f, 10)
    p, r, f = get_PRF(num_c_15, num_e_15, num_s)
    print_PRF(p, r, f, 15)


except ValueError:
    en_model.close()
en_model.close()
time_end = time.time()
print('totally cost', time_end - time_start)
