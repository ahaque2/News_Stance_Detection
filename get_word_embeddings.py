# -*- coding: utf-8 -*-
#
# Author: Amanul Haque


import corenlp
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
from gensim.models import KeyedVectors
import os
from cleantext import clean
import pprint
import sys
import itertools
import nltk
import re
import ast
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation
from autocorrect import spell
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
import corenlp
import os
import math
import statistics 

from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

os.environ['CORENLP_HOME'] = '/home/ahaque2/project/virtual_environment_1/stanfordNLP'

class get_word_embeddings:
    
    def __init__(self):
        self.features =  ['similarity', 'top5_senti', 'article_senti', 'head_senti', 'hedge_words', 'word_embeddings']
    
    #Code for Text preprocessing
    def autospell(self, text):
        """
        correct the spelling of the word.
        """
        spells = [spell(w) for w in (nltk.word_tokenize(text))]
        return " ".join(spells)
    
    def to_lower(self, text):
        """
        :param text:
        :return:
            Converted text to lower case as in, converting "Hello" to "hello" or "HELLO" to "hello".
        """
        return text.lower()
    
    def remove_numbers(self, text):
        """
        take string input and return a clean text without numbers.
        Use regex to discard the numbers.
        """
        output = ''.join(c for c in text if not c.isdigit())
        return output
    
    def remove_punct(self, text):
        """
        take string input and clean string without punctuations.
        use regex to remove the punctuations.
        """
        return ''.join(c for c in text if c not in punctuation)
    
    def remove_Tags(self, text):
        """
        take string input and clean string without tags.
        use regex to remove the html tags.
        """
        cleaned_text = re.sub('<[^<]+?>', '', text)
        return cleaned_text
    
    def sentence_tokenize(self, text):
        """
        take string input and return list of sentences.
        use nltk.sent_tokenize() to split the sentences.
        """
        sent_list = []
        for w in nltk.sent_tokenize(text):
            sent_list.append(w)
        return sent_list
    
    def word_tokenize(self, text):
        """
        :param text:
        :return: list of words
        """
        return [w for sent in nltk.sent_tokenize(text) for w in nltk.word_tokenize(sent)]
    
    def remove_stopwords(self, sentence):
        """
        removes all the stop words like "is,the,a, etc."
        """
        stop_words = stopwords.words('english')
        return ' '.join([w for w in nltk.word_tokenize(sentence) if not w in stop_words])
    
    def stem(self, text):
        """
        :param word_tokens:
        :return: list of words
        """
        
        snowball_stemmer = SnowballStemmer('english')
        stemmed_word = [snowball_stemmer.stem(word) for sent in nltk.sent_tokenize(text)for word in nltk.word_tokenize(sent)]
        return " ".join(stemmed_word)
    
    def lemmatize(self, text):
        
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized_word = [wordnet_lemmatizer.lemmatize(word)for sent in nltk.sent_tokenize(text)for word in nltk.word_tokenize(sent)]
        return " ".join(lemmatized_word)

    
    def preprocess(self, text):
    
        lower_text = self.to_lower(text)
        sentence_tokens = self.sentence_tokenize(lower_text)
        word_list = []
        for each_sent in sentence_tokens:
            lemmatizzed_sent = self.lemmatize(each_sent)
            clean_text = self.remove_numbers(lemmatizzed_sent)
            clean_text = self.remove_punct(clean_text)
            clean_text = self.remove_Tags(clean_text)
            clean_text = self.remove_stopwords(clean_text)
            word_tokens = self.word_tokenize(clean_text)
            for i in word_tokens:
                word_list.append(i)
        return word_list
            
        
    def mean_vectorizor(self, model, tokenized_sent, dim):
    
        return np.array([
                np.mean([model[w] for w in words if w in model.vocab]
                        or [np.zeros(dim)], axis=0)
                for words in tokenized_sent
            ])
                
    def get_tokenized_body_para1(self, text, dim=5):
        
        tokenized_sent = sent_tokenize(text)
        #body_text = tokenized_sent[0:dim]
        body_text = tokenized_sent
        bt = []
        [bt.extend(b.split('.')[:-1]) for b in body_text]
        return bt
        
    def get_hedge_word_counts(self, sent_list):

        f = open('hedge_words.txt')
        hedge_words = f.readlines()
        hedge_words = [h.strip('\n') for h in hedge_words]
        hedge_word_counts = []
        for sent in sent_list:
            count = 0
            for word in hedge_words:
                if word in sent:
                    count+=1
            hedge_word_counts.append(count)
        return hedge_word_counts
    
    def get_senti_score(self, comment_list):
        analyzer = SentimentIntensityAnalyzer()
        senti_score = [analyzer.polarity_scores(text) for text in comment_list]
        return senti_score  
            
        
    def get_features(self, article_id, article_body, stance_id, labels, headlines):
        
        model = KeyedVectors.load_word2vec_format('../../word_embedings/GoogleNews-vectors-negative300.bin', binary=True)
        
        df = pd.DataFrame(columns =  ['stance_id', 'label', 'word_features'])
        for bid, txt in zip(article_id, article_body):
            index = np.where(stance_id == bid)[0]
            article = np.array(self.get_tokenized_body_para1(txt))
            
            hedge_word_counts = self.get_hedge_word_counts(article)
            lab = labels.iloc[index]
            heads = headlines.iloc[index]
            heads_tokenized = [self.preprocess(h) for h in heads]
            
            head_vec = self.mean_vectorizor(model, heads_tokenized, 300)
            
            feature_vec = []
            sent_embeddings = []
            for sent in article:
                v = self.mean_vectorizor(model, [self.preprocess(sent)], 300)
                sent_embeddings.append(v)
                
            from scipy import spatial
            for h, headl in zip(head_vec, heads_tokenized):
                sim = []
                for emb in sent_embeddings:
                    sim.append(1 - spatial.distance.cosine(h, emb))
                
                sim = [0 if math.isnan(x) else x for x in sim]
                top5 = np.argsort(sim)[-5:]
                
                if(len(top5) > 0):
                    
                    sent_body = []
                    [sent_body.extend(self.preprocess(article[i])) for i in top5]
                    vec = self.mean_vectorizor(model, [sent_body], 300).tolist()[0]
                    #print("Shape of article vector ", type(vec), len(vec))
                    
                    s = statistics.mean([sim[i] for i in top5])
                    vec.append(s)
    
                    senti = self.get_senti_score(headl)
                    comp_senti = statistics.mean([a['compound'] for a in senti])
                    vec.append(comp_senti)
    
                    senti = self.get_senti_score(article)
                    comp_senti = statistics.mean([a['compound'] for a in senti])
                    vec.append(comp_senti)
    
                    top5_sent = article[top5]
                    senti = self.get_senti_score(top5_sent)
                    top5_comp_senti = statistics.mean([a['compound'] for a in senti])
                    vec.append(top5_comp_senti)
                    #print("New Vec shape ", len(vec))
                    
                    vec.append(hedge_word_counts)
                    
                else:
                    vec = np.zeros(305)
                    
                feature_vec.append(vec)
            
            df2 = pd.DataFrame(columns = ['stance_id', 'label', 'word_features'])
            df2['stance_id'] = index
            df2['label'] = np.array(lab)
            df2['word_features'] = feature_vec
    
            df = df.append(df2, ignore_index = True)
            
            #sys.exit()
    
        return df


    def get_features2(self, article_id, article_body, stance_id, labels, headlines):
        
        df = pd.DataFrame(columns = ['stance_id', 'label', 'word_features'])
        model = KeyedVectors.load_word2vec_format('../../word_embedings/GoogleNews-vectors-negative300.bin', binary=True)
            
        for bid, txt in zip(article_id, article_body):
            index = np.where(stance_id == bid)[0]
            #print("Article ", txt, type(txt))
            article = txt[0:1000]
            article = self.preprocess(article)
            
            lab = labels.iloc[index]
            heads = headlines.iloc[index]
            heads_tokenized = [self.preprocess(h) for h in heads]
            
            #print("Headlines length ", len(heads_tokenized))
            
            head_vec = self.mean_vectorizor(model, heads_tokenized, 300)
            article_vec =  self.mean_vectorizor(model, [article], 300)
            
            feature_vec = []
            #print(article_vec.shape)
            #print(head_vec.shape)
            
            from scipy import spatial
            for h in head_vec:
                s = 1 - spatial.distance.cosine(h, article_vec)
                c = np.concatenate((h, s, article_vec), axis = None)
                feature_vec.append(c)
            
            
            #print("Feature Vector length ", len(feature_vec))
              
            df2 = pd.DataFrame(columns = ['stance_id', 'label', 'word_features'])
            df2['stance_id'] = index
            df2['label'] = np.array(lab)
            df2['word_features'] = feature_vec
            
            df = df.append(df2, ignore_index = True)
    
        return df
        
    
    
    def get_Xy(self, article_body, article_stance):
        
        body = pd.read_csv(article_body)
        stance = pd.read_csv(article_stance)
        
        #print(body.shape)
        #print(stance.shape)
        
        article_id = body['Body ID']
        stance_id = stance['Body ID']
        
        article_body = body['articleBody']
        labels = stance['Stance']
        headlines = stance['Headline']
        
        df = self.get_features(article_id, article_body, stance_id, labels, headlines)
        #print("Shape of Final dataframe ", df.shape)
        df = df.drop(np.where(df.isna() == True)[0])
        #df.to_csv("combined_features/preprocessed.csv")
        y = df['label']
        X = df['word_features']
        
        return X, y
        
        