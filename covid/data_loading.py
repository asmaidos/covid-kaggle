#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:33:49 2020

@author: jkraft
"""


import os
import json
from pprint import pprint
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import nltk
import re
from time import time

from nltk.tokenize import word_tokenize
import math
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from rank_bm25 import BM25Okapi, BM25Plus # don't use BM25L, there is a mistake
import os
import torch
import numpy
from tqdm import tqdm
from transformers import *
import warnings

nltk.download('stopwords')
nltk.download('punkt')

# =============================================================================
# helper functions modified from 
# https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv
# =============================================================================



def format_name(author):
    middle_name = " ".join(author['middle'])
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])
def format_authors(authors):
    name_ls = []
    for author in authors:
        name = format_name(author)
        name_ls.append(name)
    return ", ".join(name_ls)
def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    for section, text in texts:
        texts_di[section] += text
    body = ""
    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    return body
def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []
    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    return raw_files
def generate_clean_df(all_files):
    cleaned_files = []
    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_body(file['abstract']),
            format_body(file['body_text']),
        ]
        cleaned_files.append(features)
    col_names = ['paper_id', 'title', 'authors',
                 'abstract', 'text']
    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()
    return clean_df



# split into words


def get_top_n(bm25_model, query, documents, n=5):
    """ 
    Reimplementation of the method to get the index of the top n.
    
    See https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py
    """

    scores = bm25_model.get_scores(query)
    top_n = np.argsort(scores)[::-1][:n]
    top_scores = scores[top_n]
    return top_n, top_scores


def clean_text_for_query_search(index, text_list, stopword_remove=False,
                                stemming=True,
                                stemmer=PorterStemmer()):
    """
    Clean-up the text to make a key-word query more efficient.
    
    Lower case, remove punctuation, numeric strings, stopwords,
    and finally stem.
    
    Note: Do not use before a language model
    
    Parameters
    ----------
    index : int
        Index of text in list to process, for use of vectorization
    text : String
        Text to clean
    stopword_remove : boolean
        True to remove stowords
        Article suggests that this is detrimental
    stemming : boolean
        True for stemming the text
        Article suggests that this is only useful with a very weak stemmer   
    stemmer : NLTK stemmer
        Stemmer to use
        
    Returns
    -------
    cleaned_text : string
        Processed text
    """
    # retrieve element from list to use tqdm
    text = text_list[index]
    
    punc_table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))
    
    def process_tokens(token):
        """ Vectorize token processing."""
        token_low = token.lower()
        stripped = token_low.translate(punc_table)
        alpha = stripped if not stripped.isnumeric() else ''
        # remove stop words filtering as suggested in article
        if stopword_remove:
            alpha = alpha if not alpha in stop_words else ''
        # also disable stemmer! as suggested in article
        if stemming:
            alpha = stemmer.stem(alpha)
        
        return alpha
    
    concat_doc = list(map(lambda x: process_tokens(x), word_tokenize(text)))
    cleaned_doc = ' '.join([w for w in concat_doc if not w == ''])
    
    return cleaned_doc


	
os.chdir('/home/jkraft/Dokumente/Kaggle/')

# some titles and abstract are empty in the data
data_df = pd.read_csv("./1000_doc_df.csv")

# =============================================================================
# Try BM25
# =============================================================================


# in the implementation!


task_text = ["What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?"]
quest_filename = "./question1.txt"

def read_question(task_text, quest_filename):
    # get task text
    with open(quest_filename) as f: 
        quest_text = task_text + f.readlines()
    return quest_text

def search_corpus_for_question(quest_text, data_df, model=BM25Plus, top_n=10,
                               col='cleaned_text'):
    """
    Clean-up the text to make a key-word query more efficient.
    
    Lower case, remove punctuation, numeric strings, stopwords,
    and finally stem.
    
    Note: Do not use before a language model
    
    Parameters
    ----------
    quest_text : List of string
        Lines of the questions + task.
    data_df : Pandas dataframe
        Dataframe with corpus text
    model: BM25 model
        Model to use
    top_n: int
        quantity of results to return 
    col: string
        column to search on
        
    Returns
    -------
    indices : list of int
        Indices of answers in the input dataframe
    scores : list of float
        scores of the top documents
    flat_query : list of string
        Prepared query text
    """
    # create BM25 model
    corpus = data_df[col]
    tokenized_corpus = [str(doc).split(" ") for doc in corpus]
    bm25 = model(tokenized_corpus)
    
    # prepare query
    cleaned_query = list(map(
            lambda x: clean_text_for_query_search(x, quest_text), trange(len(quest_text))))
    flat_query = " ".join(map(str, cleaned_query))
    tokenized_query = list(flat_query.split(" "))
    # search
    indices, scores = get_top_n(bm25, tokenized_query, corpus, n=top_n)

    return indices, scores, flat_query

# TO DO: try averaging the last 4 layers embeddings
def extract_scibert(text, tokenizer, model):
    text_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    text_words = tokenizer.convert_ids_to_tokens(text_ids[0])[1:-1]

    n_chunks = int(numpy.ceil(float(text_ids.size(1)) / 510))
    states = []
    class_states = []
    for ci in range(n_chunks):
        text_ids_ = text_ids[0, 1 + ci * 510:1 + (ci + 1) * 510]
        text_ids_ = torch.cat([text_ids[0, 0].unsqueeze(0), text_ids_])
        if text_ids[0, -1] != text_ids[0, -1]:
            text_ids_ = torch.cat([text_ids_, text_ids[0, -1].unsqueeze(0)])

        with torch.no_grad():
            state = model(text_ids_.unsqueeze(0))[0]
            # stock all embeddings and the CLS embeddings
            state = state[:, 1:-1, :]
            class_state = state[:, 0, :]
        states.append(state)
        class_states.append(class_state)

    state = torch.cat(states, axis=1)
    class_state = torch.cat(class_states, axis=1)

    return text_ids, text_words, state[0], class_state[0]


def cross_match(state1, state2, use_CLS=False):
    if not use_CLS:
        sim = torch.cosine_similarity(torch.mean(state1, 0), torch.mean(state2, 0), dim=0)
    else:
        sim = torch.cosine_similarity(state1, state2, dim=0)
    sim = sim.numpy()
    return sim


# quest_text = read_question(task_text, quest_filename)

# consider only papers about the coronavirus
 # don't search for the term 'coronavirus' to avoid animal diseases
quest_text = ['covid19', 'covid-19', 'cov-19', 'ncov-19', 'sarscov2', '2019novel',
              'SARS-CoV-2', '2019-nCoV', '2019nCoV', 'SARSr-CoV']
              # + ['Wuhan']

indices, scores, _ = search_corpus_for_question(
    quest_text, data_df, BM25Okapi, len(data_df), 'cleaned_text')
# select only the documents containing the coronavirus terms in the abstract
# almost all documents contain the term in the text...
contain_coron_in = np.array(indices)[scores>0]
data_df_red = data_df.iloc[contain_coron_in, :].copy()
data_df_red = data_df_red.reset_index(drop=True)

data_df_red_title = data_df_red['cleaned_title']

# search the question, use only keywords
task_text = ['COVID-19 risk factors? epidemiological studies']
quest_text = ['COVID-19 risk factors? epidemiological studies',
 'risks factors',
 '    Smoking, pre-existing pulmonary disease',
 '    Co-infections co-existing respiratory viral infections virus transmissible virulent co-morbidities',
 '    Neonates pregnant',
 '    Socio-economic behavioral factors economic impact virus differences']


task_text = ['Neonates and pregnant women.']

quest_text = read_question(task_text, quest_filename)
# remove \n, but has no impact, this is already done somewhere else
# quest_text = list(map(lambda x: str(x).strip(), quest_text))
model = BM25Okapi
indices, scores, quest = search_corpus_for_question(
    quest_text, data_df_red, model, len(data_df_red), 'cleaned_text')
# remove again docs without keywords if searched with Okapi BM25
if model == BM25Okapi:
    contain_keyword = np.array(indices)[scores>0]
    answers = data_df_red.iloc[contain_keyword, :].copy()
    answers['scores_BM25'] = scores[scores>0]
else:
    answers = data_df_red.iloc[indices, :].copy()
    answers['scores_BM25'] = scores
answers = answers.sort_values(['scores_BM25'], ascending=False)
answers = answers.reset_index(drop=True)

ans = answers[['scores_BM25', 'title', 'abstract', 'text']][:20]

# TO DO: add titles to abstract, as some papers lack abstracts



warnings.simplefilter(action='ignore', category=FutureWarning)

tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed', do_lower_case=False)
model = AutoModel.from_pretrained('monologg/biobert_v1.1_pubmed')

# tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
# model = AutoModelWithLMHead.from_pretrained("bert-large-cased")

# tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
# model = AutoModelWithLMHead.from_pretrained("bert-large-cased")


# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
# model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
#
# tokenizer = AutoTokenizer.from_pretrained("albert-xxlarge-v1")
# model = AutoModelWithLMHead.from_pretrained("albert-xxlarge-v1")
#
#
# tokenizer = AutoTokenizer.from_pretrained("albert-large-v1")
# model = AutoModelWithLMHead.from_pretrained("albert-large-v1")
#
#


t = time()
flat_query = ''.join(quest_text)

flat_query = 'COVID-19 risk factors? Data on potential risks factors. Smoking, pre-existing pulmonary disease. Co-infections (determine whether co-existing respiratory / viral infections make the virus more transmissible or virulent) and other co-morbidities. Neonates and pregnant women. Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.'

flat_query = 'COVID-19 risk factors? Neonates and pregnant women.'

flat_query = 'Smoking, pre-existing pulmonary disease'

flat_query = 'Neonates and pregnant women.'

query_ids, query_words, query_state, query_class_state = extract_scibert(flat_query, tokenizer, model)
print((time()-t)*1000)
print(len(query_words))
# print(query_state)

# remove rows without titles / abstracts
ans_red = ans.dropna()
ans_red = ans_red.reset_index(drop=True)
sim_scores = []

use_CLS = True
# compute similarity scores
for text in tqdm(ans_red['abstract']):
    text_ids, text_words, state, class_state = extract_scibert(text, tokenizer, model)
    if use_CLS:
        sim_score = cross_match(query_class_state, class_state, True) # try cosine on CLS tokens
    else:
        sim_score = cross_match(query_state, state, False)
    sim_scores.append(sim_score)


# Select the index of top 5 paragraphs with highest relevance
rel_index = np.flip(numpy.argsort(sim_scores))[:5]
end_ans = ans_red.iloc[rel_index, :]
end_ans = end_ans.reset_index(drop=True)
end_ans['score_ML'] = np.array(sim_scores)[rel_index]
# reorder columns
end_ans = end_ans[['score_ML', 'scores_BM25', 'title', 'abstract']]

# TO DO: try more articles

# tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
# model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
# input_context = 'I#m starting to play football'
# input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
# outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=2)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
# for i in range(3): #  3 output sequences were generated
#     print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))