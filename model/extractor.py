#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19
import nltk
from model import input_representation

#GRAMMAR1 is the general way to extract NPs

GRAMMAR1 = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR2 = """  NP:
        {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR3 = """  NP:
        {<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""


def extract_candidates(tokens_tagged, no_subset=False):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """
    np_parser = nltk.RegexpParser(GRAMMAR1)  # Noun phrase parser
    keyphrase_candidate = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ' '.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length
            keyphrase_candidate.append((np, start_end))

        else:
            count += 1

    return keyphrase_candidate

# if __name__ == '__main__':
#     #This is an example.
#     sent17 = "NuVox shows staying power with new cash, new market Who says you can't raise cash in today's telecom market? NuVox Communications positions itself for the long run with $78.5 million in funding and a new credit facility"
#     sent10 = "This paper deals with two questions: Does social capital determine innovation in manufacturing firms? If it is the case, to what extent? To deal with these questions, we review the literature on innovation in order to see how social capital came to be added to the other forms of capital as an explanatory variable of innovation. In doing so, we have been led to follow the dominating view of the literature on social capital and innovation which claims that social capital cannot be captured through a single indicator, but that it actually takes many different forms that must be accounted for. Therefore, to the traditional explanatory variables of innovation, we have added five forms of structural social capital (business network assets, information network assets, research network assets, participation assets, and relational assets) and one form of cognitive social capital (reciprocal trust). In a context where empirical investigations regarding the relations between social capital and innovation are still scanty, this paper makes contributions to the advancement of knowledge in providing new evidence regarding the impact and the extent of social capital on innovation at the two decisionmaking stages considered in this study"
#
#     input=input_representation.InputTextObj(sent10,is_sectioned=True,database="Inspec")
#     keyphrase_candidate= extract_candidates(input)
#     for kc in keyphrase_candidate:
#         print(kc)