#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19

from model import extractor
from nltk.corpus import stopwords
stopword_dict = set(stopwords.words('english'))
# from stanfordcorenlp import StanfordCoreNLP
# en_model = StanfordCoreNLP(r'E:\Python_Files\stanford-corenlp-full-2018-02-27',quiet=True)
class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, en_model, text=""):
        """
        :param is_sectioned: If we want to section the text.
        :param en_model: the pipeline of tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        self.considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}

        self.tokens = []
        self.tokens_tagged = []
        self.tokens = en_model.word_tokenize(text)
        self.tokens_tagged = en_model.pos_tag(text)
        assert len(self.tokens) == len(self.tokens_tagged)
        for i, token in enumerate(self.tokens):
            if token.lower() in stopword_dict:
                self.tokens_tagged[i] = (token, "IN")
        self.keyphrase_candidate = extractor.extract_candidates(self.tokens_tagged, en_model)

# if __name__ == '__main__':
#     text = "Adaptive state feedback control for a class of linear systems with unknown bounds of uncertainties The problem of adaptive robust stabilization for a class of linear time-varying systems with disturbance and nonlinear uncertainties is considered. The bounds of the disturbance and uncertainties are assumed to be unknown, being even arbitrary. For such uncertain dynamical systems, the adaptive robust state feedback controller is obtained. And the resulting closed-loop systems are asymptotically stable in theory. Moreover, an adaptive robust state feedback control scheme is given. The scheme ensures the closed-loop systems exponentially practically stable and can be used in practical engineering. Finally, simulations show that the control scheme is effective"
#     ito = InputTextObj(en_model, text)
#     print("OK")