from util import *
from get_word_embeddings import *
import random
import tensorflow as tf
import numpy as np
import sys
import pandas as pd
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import statistics

tf.compat.v1.disable_eager_execution()

# Initialise hyperparameters
r = random.Random()
lim_unigram = 5000
target_size = 4
hidden_size = 100
train_keep_prob = 0.6
l2_alpha = 0.00001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 500
epochs = 90

def reshape_arrays(trainX, trainY, testX, testY):
    
    trainX = np.array([x for x in trainX])
    trainY = np.array([y for y in trainY])
    
    testX = np.array([x for x in testX])
    testY = np.array([y for y in testY])
    
    return trainX, trainY, testX, testY

def change_labels_to_numeric(labels):
    
    y = np.array([None] * labels.shape[0])
    
    y[np.where(labels == 'agree')[0]] = 0
    y[np.where(labels == 'disagree')[0]] = 1
    y[np.where(labels == 'discuss')[0]] = 2
    y[np.where(labels == 'unrelated')[0]] = 3
    
    return y
   

def load_pretrained_vectors(X_train, X_test, y_train, y_test):
    
    trainX, trainY = np.load(X_train), np.load(y_train)
    testX, testY = np.load(X_test), np.load(y_test)
    
    trainX, trainY, testX, testY = reshape_arrays(trainX, trainY, testX, testY)
    
    feature_size = len(trainX[0])
    n_train = trainX.shape[0]
    
    return trainX, trainY, testX, testY, n_train, feature_size

def generate_word_embeddings(data_source, file_train_bodies, file_train_instances, file_test_bodies, file_test_instances):
    
    gwe = get_word_embeddings()
    trainX, trainY = gwe.get_Xy(file_train_bodies, file_train_instances)
    trainY = change_labels_to_numeric(trainY)
    feature_size = len(trainX[0])
    n_train = trainX.shape[0]
    
    testX, testY = gwe.get_Xy(file_test_bodies, file_test_instances)
    testY = change_labels_to_numeric(testY)
    
    trainX, trainY, testX, testY = reshape_arrays(trainX, trainY, testX, testY)
         
    np.save(data_source + "/trainX.npy", np.array(trainX))
    np.save(data_source + "/trainY.npy", np.array(trainY))
    np.save(data_source + "/testX.npy", np.array(testX))
    np.save(data_source + "/testY.npy", np.array(testY))   

    return trainX, trainY, testX, testY, n_train, feature_size
    

def generate_tf_idf(data_source, file_train_instances, file_train_bodies, file_test_instances, file_test_bodies):       
    
    # Load data sets
    raw_train = FNCData(file_train_instances, file_train_bodies)
    raw_test = FNCData(file_test_instances, file_test_bodies)
    n_train = len(raw_train.instances)
    
    # Process data sets
    trainX, trainY, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
        pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
    feature_size = len(trainX[0])
    testX = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)    
    testY = np.array(pd.read_csv(file_test_instances)['Stance'])
    testY = change_labels_to_numeric(testY)
    
    return trainX, trainY, testX, testY, n_train, feature_size


def get_data_source(data_source_id):
    
    data_source = "processed_word_embedding/"
    
    if(data_source_id == 1):
        data_source = "processed_word_embedding/"
    elif(data_source_id == 2):
        data_source = "summarized_word_embedding/"
    elif(data_source_id == 3):
        data_source = "word_embedding_with_no_preprocessing/"
        
    return data_source
        
def get_pretrain_word_vectors_files(data_source):
    
    X_train = data_source + "trainX.npy"
    y_train = data_source + "trainY.npy"
    X_test = data_source + "testX.npy"
    y_test = data_source + "testY.npy"     
    
    return X_train, y_train, X_test, y_test
    

def get_summarized_data():
    
    #Datasets with summarized text for news articles
    file_train_instances = "data/train_stances.csv"
    file_train_bodies = "data/train_summarized_bodies.csv"
    file_test_instances = "data/train_stances.csv"
    file_test_bodies = "data/test_summarized_bodies.csv"
    
    return file_train_instances, file_train_bodies, file_test_instances, file_test_bodies


def get_data():
    
    #Datasets
    file_train_instances = "data/train_stances.csv"
    file_train_bodies = "data/train_bodies.csv"
    file_test_instances = "data/test_stances.csv"
    file_test_bodies = "data/test_bodies.csv"
    
    return file_train_instances, file_train_bodies, file_test_instances, file_test_bodies

  
file_predictions = 'data/predictions_test.csv'

#Flags to change configuration to run different approaches
pretrained_flag = 1     #Change to 0 to generate word-embedding instead of using pre-generated 
word2vec_flag = 1       #Change to 1 to run with Tf-Idf instead of word-embeddings
data_source_id = 1      #Choose appropriate data_source 
summarized_data_flag = 0    #Change this flag to 1 if you wish to use the summarized text instead of full text dataset

data_source = get_data_source(data_source_id)
    
if(pretrained_flag == 1):
    X_train, y_train, X_test, y_test = get_pretrain_word_vectors_files(data_source)
else:
    if(summarized_data_flag == 0):
        file_train_instances, file_train_bodies, file_test_instances, file_test_bodies  = get_data()
    else:
        file_train_instances, file_train_bodies, file_test_instances, file_test_bodies  = get_summarized_data()
    
if(word2vec_flag != 1):
    trainX, trainY, testX, testY, n_train, feature_size = generate_tf_idf(data_source, file_train_instances, file_train_bodies, file_test_instances, file_test_bodies)

elif(pretrained_flag == 1):    
    trainX, trainY, testX, testY, n_train, feature_size = load_pretrained_vectors(X_train, X_test, y_train, y_test)
    
else:
    trainX, trainY, testX, testY, n_train, feature_size = generate_word_embeddings(data_source, file_train_bodies, file_train_instances, file_test_bodies, file_test_instances)

print("X_train ", trainX.shape)
print("Y_train ", trainY.shape)
print("X_test ", testX.shape)
print("Y_test ", testY.shape)
print("Number of training instances ", n_train)
print("Feature_size ", feature_size)

unique, counts = np.unique(trainY, return_counts=True)
print("Train Dataset class frequencies ", dict(zip(unique, counts)))


unique, counts = np.unique(testY, return_counts=True)
print("Test Dataset class frequencies ", dict(zip(unique, counts)))

# Create placeholders
features_pl = tf.compat.v1.placeholder(tf.float32, [None, feature_size], 'features')
stances_pl = tf.compat.v1.placeholder(tf.int64, [None], 'stances')
keep_prob_pl = tf.compat.v1.placeholder(tf.float32)

# Infer batch size
batch_size = tf.shape(input=features_pl)[0]

# Define multi-layer perceptron
hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), rate=1 - (keep_prob_pl))
logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size), rate=1 - (keep_prob_pl))
logits = tf.reshape(logits_flat, [batch_size, target_size])

# Define L2 loss
tf_vars = tf.compat.v1.trainable_variables()
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

# Define overall loss
loss = tf.reduce_sum(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(labels = stances_pl, logits = logits) + l2_loss)

# Define prediction
softmaxed_logits = tf.nn.softmax(logits)
predict = tf.argmax(input=softmaxed_logits, axis=1)


# Load pretrained model when using TF-IDF
if word2vec_flag != 1 and pretrained_flag == 1:
    with tf.compat.v1.Session() as sess:        
        load_model(sess)        
        # Predict
        test_feed_dict = {features_pl: testX, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)


# Train model
else:

    # Define optimiser
    opt_func = tf.compat.v1.train.AdamOptimizer(learn_rate)
    grads, _ = tf.clip_by_global_norm(tf.gradients(ys=loss, xs=tf_vars), clip_ratio)
    opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

    # Perform training
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(epochs):
            total_loss = 0
            indices = list(range(n_train))
            r.shuffle(indices)

            for i in range(n_train // batch_size_train):
                batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
                batch_features = [trainX[i] for i in batch_indices]
                batch_stances = [trainY[i] for i in batch_indices]

                batch_feed_dict = {features_pl: batch_features, stances_pl: batch_stances, keep_prob_pl: train_keep_prob}
                _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
                total_loss += current_loss


        # Predict
        test_feed_dict = {features_pl: testX, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)


# Save predictions
save_predictions(test_pred, file_predictions)
print(test_pred)
print(testY)
print(type(test_pred), test_pred.shape, type(testY), testY.shape)
print(set(test_pred), set(testY))

print(testY.shape, test_pred.shape)

unique, counts = np.unique(testY, return_counts=True)
print(dict(zip(unique, counts)))

unique, counts = np.unique(test_pred, return_counts=True)
print(dict(zip(unique, counts)))

y_true = [int(y) for y in testY]
y_pred = [int(y) for y in test_pred]  

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

print("Precision : ", precision_score(y_true, y_pred, average='weighted'))
print("Recall : ", recall_score(y_true, y_pred, average='weighted'))
