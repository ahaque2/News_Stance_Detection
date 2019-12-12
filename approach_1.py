# -*- coding: utf-8 -*-
#
# Author: Amanul Haque

import pandas as pd
import numpy as np
import sys
from gensim.models import KeyedVectors
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation
from sklearn.svm import LinearSVC
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import math
import statistics 
from scipy import spatial

from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def load_dataset(body_file, stance_file):

    data_body = pd.read_csv(body_file)
    data_stance = pd.read_csv(stance_file)
    
    article_id = data_body['Body ID']
    stance_id = data_stance['Body ID']
    
    article_body = data_body['articleBody']
    labels = data_stance['Stance']
    headlines = data_stance['Headline']
    
    return article_id, article_body, stance_id, labels, headlines


def reshape_arrays(trainX, trainY, testX, testY):
    
    trainX = np.array([x for x in trainX])
    trainY = np.array([y for y in trainY])
    
    testX = np.array([x for x in testX])
    testY = np.array([y for y in testY])
    
    return trainX, trainY, testX, testY


def remove_nans(X_train, y_train, X_test, y_test):
    
    Xtrain_nan = np.isnan(X_train)
    ytrain_nan = np.isnan(y_train)
    Xtest_nan = np.isnan(X_test)
    ytest_nan = np.isnan(y_test)

    nan_index = []
    nan_index.extend(np.where(Xtrain_nan == True)[0])
    nan_index.extend(np.where(ytrain_nan == True)[0])
    nan_index.extend(np.where(Xtest_nan == True)[0])
    nan_index.extend(np.where(ytest_nan == True)[0])

    X_train = np.delete(X_train, nan_index, axis = 0)
    y_train = np.delete(y_train, nan_index, axis = 0)
    X_test = np.delete(X_test, nan_index, axis = 0)
    y_test = np.delete(y_test, nan_index, axis = 0)
    
    return X_train, y_train, X_test, y_test


def change_labels_to_numeric(labels):
    
    y = np.array([None] * labels.shape[0])
    
    y[np.where(labels == 'agree')[0]] = 0
    y[np.where(labels == 'disagree')[0]] = 1
    y[np.where(labels == 'discuss')[0]] = 2
    y[np.where(labels == 'unrelated')[0]] = 3
    
    return y

def load_pregenerated_features_vectors():
    
    print("Loading data")
    
    data_source = "baseline_features/"
    trainX = data_source + "trainX.npy"
    trainY = data_source + "trainY.npy"
    testX = data_source + "testX.npy"
    testY = data_source + "testY.npy" 
    
    X_train, y_train = np.load(trainX), np.load(trainY)
    X_test, y_test = np.load(testX), np.load(testY)   
    
    y_train = change_labels_to_numeric(y_train)
    y_test = change_labels_to_numeric(y_test)
    
    X_train, y_train, X_test, y_test = reshape_arrays(X_train, y_train, X_test, y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    X_train, y_train, X_test, y_test = remove_nans(X_train, y_train, X_test, y_test)
    X_train, y_train, X_test, y_test = reshape_arrays(X_train, y_train, X_test, y_test)  
    
    return X_train, y_train, X_test, y_test


def mean_vectorizor(model, tokenized_sent, dim):
    
    return np.array([
            np.mean([model[w] for w in words if w in model.vocab]
                    or [np.zeros(dim)], axis=0)
            for words in tokenized_sent
        ])
            
            
def get_tokenized_body_para1(text, dim=5):
    
    tokenized_sent = sent_tokenize(text)
    #body_text = tokenized_sent[0:dim]
    body_text = tokenized_sent
    bt = []
    [bt.extend(b.split('.')[:-1]) for b in body_text]
    return bt

def get_senti_score(comment_list):
    analyzer = SentimentIntensityAnalyzer()
    senti_score = [analyzer.polarity_scores(text) for text in comment_list]
    return senti_score  
            
def to_lower(text):
    """
    :param text:
    :return:
        Converted text to lower case as in, converting "Hello" to "hello" or "HELLO" to "hello".
    """
    return text.lower()

def remove_numbers(text):
    """
    take string input and return a clean text without numbers.
    Use regex to discard the numbers.
    """
    output = ''.join(c for c in text if not c.isdigit())
    return output

def remove_punct(text):
    """
    take string input and clean string without punctuations.
    use regex to remove the punctuations.
    """
    return ''.join(c for c in text if c not in punctuation)

def remove_Tags(text):
    """
    take string input and clean string without tags.
    use regex to remove the html tags.
    """
    cleaned_text = re.sub('<[^<]+?>', '', text)
    return cleaned_text

def sentence_tokenize(text):
    """
    take string input and return list of sentences.
    use nltk.sent_tokenize() to split the sentences.
    """
    sent_list = []
    for w in nltk.sent_tokenize(text):
        sent_list.append(w)
    return sent_list

def word_tokenize(text):
    """
    :param text:
    :return: list of words
    """
    return [w for sent in nltk.sent_tokenize(text) for w in nltk.word_tokenize(sent)]

def remove_stopwords(sentence):
    """
    removes all the stop words like "is,the,a, etc."
    """
    stop_words = stopwords.words('english')
    return ' '.join([w for w in nltk.word_tokenize(sentence) if not w in stop_words])

def stem(text):
    """
    :param word_tokens:
    :return: list of words
    """
    
    snowball_stemmer = SnowballStemmer('english')
    stemmed_word = [snowball_stemmer.stem(word) for sent in nltk.sent_tokenize(text)for word in nltk.word_tokenize(sent)]
    return " ".join(stemmed_word)

def lemmatize(text):
    
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word)for sent in nltk.sent_tokenize(text)for word in nltk.word_tokenize(sent)]
    return " ".join(lemmatized_word)


def preprocess(text):

    lower_text = to_lower(text)
    sentence_tokens = sentence_tokenize(lower_text)
    word_list = []
    for each_sent in sentence_tokens:
        lemmatizzed_sent = lemmatize(each_sent)
        clean_text = remove_numbers(lemmatizzed_sent)
        clean_text = remove_punct(clean_text)
        clean_text = remove_Tags(clean_text)
        clean_text = remove_stopwords(clean_text)
        word_tokens = word_tokenize(clean_text)
        for i in word_tokens:
            word_list.append(i)
    return word_list

def get_hedge_word_counts(sent_list, hedge_words):
    hedge_word_counts = []
    for sent in sent_list:
        count = 0
        for word in hedge_words:
            if word in sent:
                count+=1
        hedge_word_counts.append(count)
    return hedge_word_counts

def get_features(article_id, article_body, stance_id, labels, headlines):
    
    df = pd.DataFrame(columns = ['stance_id', 'similarity', 'label'])
    for bid, txt in zip(article_id, article_body):
        index = np.where(stance_id == bid)[0]
        article = np.array(get_tokenized_body_para1(txt))
        
        
        hedge_word_counts = get_hedge_word_counts(article, hedge_words)
        lab = labels.iloc[index]
        heads = headlines.iloc[index]
        heads_tokenized = [preprocess(h) for h in heads]
        
        head_vec = mean_vectorizor(model, heads_tokenized, 300)
        
        similarity = []
        sentiment = []
        head_senti = []
        top5_senti = []
        for h, headl in zip(head_vec, heads_tokenized):
            sim = []
            for sent in article:
                v = mean_vectorizor(model, [preprocess(sent)], 300)
                sim.append(1 - spatial.distance.cosine(h, v))
                    
            sim = [0 if math.isnan(x) else x for x in sim]
            top5 = np.argsort(sim)[-5:]
            
            if(len(top5) > 0):
            
                s = statistics.mean([sim[i] for i in top5])
                similarity.append(s)

                senti = get_senti_score(headl)
                comp_senti = statistics.mean([a['compound'] for a in senti])
                head_senti.append(comp_senti)

                senti = get_senti_score(article)
                comp_senti = statistics.mean([a['compound'] for a in senti])
                sentiment.append(comp_senti)

                top5_sent = article[top5]
                senti = get_senti_score(top5_sent)
                top5_comp_senti = statistics.mean([a['compound'] for a in senti])
                top5_senti.append(top5_comp_senti)
            
            else:
                similarity.append(0)
                head_senti.append(0)
                sentiment.append(0)
                top5_senti.append(0)

        df2 = pd.DataFrame(columns = ['stance_id', 'label', 'similarity', 'top5_senti', 'article_senti', 'head_senti', 'hedge_words'])
        df2['stance_id'] = index
        df2['similarity'] = similarity
        df2['label'] = np.array(lab)
        df2['top5_senti'] = top5_senti
        df2['article_senti'] = sentiment
        df2['head_senti'] = head_senti
        df2['hedge_words'] = sum(hedge_word_counts)

        df = df.append(df2, ignore_index = True)

    return df

def get_Xy(df, features):
    
    df = df.drop(np.where(df.isna() == True)[0])
    y = df['label']
    X = df[features]
    
    return X, y

model = None

f = open('hedge_words.txt')
hedge_words = f.readlines()
hedge_words = [h.strip('\n') for h in hedge_words]

pretrained_flag = 1

if(pretrained_flag != 1): 
    
    print("Loading Data ")
    model = KeyedVectors.load_word2vec_format('../../word_embedings/GoogleNews-vectors-negative300.bin', binary=True)

    train_article_id, train_article_body, train_stance_id, train_labels, train_headlines = load_dataset("data/train_bodies.csv", "data/train_stances.csv")
    test_article_id, test_article_body, test_stance_id, test_labels, test_headlines = load_dataset("data/test_bodies.csv", "data/test_stances.csv")
        
    df_train = get_features(train_article_id, train_article_body, train_stance_id, train_labels, train_headlines)
    df_test = get_features(test_article_id, test_article_body, test_stance_id, test_labels, test_headlines)
    print(df_train.shape, df_test.shape)
    
    features = ['similarity', 'hedge_words']
    X_train, y_train = get_Xy(df_train, features)
    
    X_test, y_test = get_Xy(df_test, features)

else:
    X_train, y_train, X_test, y_test = load_pregenerated_features_vectors()

print("Training set shapes : ", X_train.shape, y_train.shape)
print("Testing set shapes ", X_test.shape, y_test.shape)
unique, counts = np.unique(y_test, return_counts=True)
print("Data distribution in test set ", dict(zip(unique, counts)))

#clf = tree.DecisionTreeClassifier()
#clf = MultinomialNB()
#clf = neighbors.KNeighborsClassifier(9)
clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
#clf = svm.SVC(gamma='scale')
#clf = LinearSVC(random_state=0, tol=1e-5)

#clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

unique, counts = np.unique(y_test, return_counts=True)
print(dict(zip(unique, counts)))

y_true = [y for y in y_test]
y_pred = [y for y in y_pred]  

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

print("Precision : ", precision_score(y_true, y_pred, average='weighted'))
print("Recall : ", recall_score(y_true, y_pred, average='weighted'))