#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/7/29

from bert_serving.client import BertClient
import numpy as np
class WordEmbeddings():
    """
        Concrete class of @EmbeddingDistributor using ELMo
        https://allennlp.org/elmo

    """

    def __init__(self,N=768):

        self.bert = BertClient()
        self.N = N

    def get_tokenized_words_embeddings(self, sents_tokened):
        """
        @see EmbeddingDistributor
        :param tokenized_sents: list of tokenized words string (sentences/phrases)
        :return: ndarray with shape (len(sents), dimension of embeddings)
        """
        bert_embeddings=[]
        for i in range(0, len(sents_tokened)):
            length = len(sents_tokened[i])
            b_e = np.zeros((1, length, self.N))
            b_e[0]=self.bert.encode(sents_tokened[i])
            bert_embeddings.append(b_e)

        return np.array( bert_embeddings)


if __name__ == '__main__':
    Bert=WordEmbeddings()
    sent_tokens=[['I',"love","Rock","and","R","!"],['I',"love","Rock","and","R","!"]]
    embs=Bert.get_tokenized_words_embeddings(sent_tokens)
    print(embs)
