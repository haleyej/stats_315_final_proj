# STATS 315, Statistics and Artifical Intelligence, at the University of Michigan
# Contains helper functions for my final project 
# intended to help clean up the notebook where I'm training my models
# Haley Johnson

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf 
import os 

def pad_inputs(s, max_len):
    '''
    Pads inputs so all sequences 
    will be the same length
    '''
    if len(s) < max_len:
        return [0] * (max_len - len(s)) + s
    elif len(s) > max_len:
        return s[:max_len]
    else:
        return s

def get_embeddings(glove, corpus, vocab_size, embedding_size):
    '''
    Takes in corpus and glove embedding layer

    Returns embeddings for each word in corpus 
    '''
    unique_words = sorted(list(set(corpus)))
    embedding_weights = np.zeros((vocab_size, embedding_size))
    for idx, word in enumerate(unique_words):
        if word in glove:
            embedding_weights[idx + 1, :] = glove[word]
    return embedding_weights

def get_glove_embedding_input(X_train_tokens, y_train, X_test, y_test, X_val, y_val, tokenizer, MAX_LEN, NUM_CLASSES):
    '''
    Takes in test, train and validation data
    Prepares it for model using glove embedding
    Makes call to pad_inputs
    
    note that X_train should already be tokenized (it's just the way I wrote this I know it's dumb)
    (but we need to fit the tokenizer on X_train bc ~leakage~) 

    Returns test, train and validation data
    '''
    y_train_input = tf.keras.utils.to_categorical(y_train-1, num_classes = NUM_CLASSES)
    X_train_input = np.array(X_train_tokens)

    X_val_input = tokenizer.texts_to_sequences(X_val)
    X_val_input = np.array([pad_inputs(s, MAX_LEN) for s in X_val_input])
    y_val_input = tf.keras.utils.to_categorical(y_val - 1, num_classes = NUM_CLASSES)

    X_test_input = tokenizer.texts_to_sequences(X_test)
    X_test_input = np.array([pad_inputs(s, MAX_LEN) for s in X_test_input])
    y_test_input = tf.keras.utils.to_categorical(y_test-1, num_classes = NUM_CLASSES)
    
    return (X_train_input, y_train_input, X_test_input, y_test_input, X_val_input, y_val_input)


def get_bert_encodings(s, tokenizer, MAX_LEN):
    '''
    Ads start/end tags, splits Tweet into tokens and pads them 
    Note that the tokenizer should be a BERT tokenizer 

    Makes call to pad_inputs function 

    Return padded token ids
    '''
    s = '[CLS]' + s + '[SEP]'
    s = s[:MAX_LEN - 2]
    tokens = tokenizer.tokenize(s)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    padded_tokens = pad_inputs(tokens, MAX_LEN)
    padded_token_ids = pad_inputs(token_ids, MAX_LEN)
    return np.array(padded_token_ids)


def get_token_mask(s, MAX_LEN):
    '''
    Returns a mask of 1's and 0's that indicates
    what part of the vector is padding and what
    part is word

    Starts with 0s since our vectors are foward-padded
    '''
    s = np.split(s, np.argwhere(s == 0.0).flatten()[1:])
    s = s[-1][1:]
    return np.array([0] * (MAX_LEN - len(s)) + [1] * len(s))


def get_segment_ids(s, MAX_LEN):
    '''
    Returns segment ids  
    Note: This code considers the entire Tweet to be one sentence
    Segment id's will all be 1
    '''
    return np.array([1] * MAX_LEN)


def get_bert_inputs(X, tokenizer, MAX_LEN):
    '''
    Calls get_bert_encodings, get_token_mask 
    and get_segment_ids to produce the 3 kind of
    inputs we need for the BERT embedding layer
    '''
    tokens = X.apply(get_bert_encodings, args = (tokenizer, MAX_LEN))
    masks = tokens.apply(get_token_mask, args = [MAX_LEN])
    segment_ids = tokens.apply(get_segment_ids, args = [MAX_LEN])

    tokens = [np.array(i) for i in tokens]
    masks = [np.array(i) for i in masks]
    segment_ids = [np.array(i) for i in segment_ids]

    return (np.array(tokens), np.array(masks), np.array(segment_ids))


def sparse2tensor(X):
        '''
        Converts sparse matrix numpy to Tensorflow tensor 

        Credit to this stack overflow post: 
        https://stackoverflow.com/questions/42560245/use-coo-matrix-in-tensorflow

        ^code is not copied directly from post, just heavily inspired by it :) 
        '''
        X = X.tocoo()
        idx = np.mat([X.row, X.col]).T
        return tf.sparse.reorder(tf.SparseTensor(idx, X.data, X.shape))


def plot_confusion_matrix(confusion_matrix, plot_title, path = "../plots"):
    '''
    Takes in confusion matrix

    Displays and saves plot showing model performance/predictions
    Returns nothing 
    '''
    normalized_confusion_matrix = confusion_matrix / confusion_matrix.astype(float).sum(axis = 1)[:, np.newaxis]

    f = sns.heatmap(normalized_confusion_matrix, annot = True, fmt = '.2f')

    f.set(title = 'Normalized Confusion Matrix of Climate Change Tweets', 
            xlabel = 'Predicted Sentiment', 
            ylabel = 'True Sentiment')

    f.set_yticklabels(['Denial', 'Neutral', 'Positive', 'News'])
    f.set_xticklabels(['Denial', 'Neutral', 'Positive', 'News'])

    plt.savefig(os.path.join(path, f"{plot_title}.png"))

    plt.show()


def plot_history(epoch_range, model_history, plot_title, path = "../plots"):
    '''
    Takes in model history 

    Displays and saves plot of accuracy vs loss for each epoch
    Returns nothing
    '''
    training_loss = model_history.history['loss']
    training_accuracy = model_history.history['accuracy']

    val_loss = model_history.history['val_loss']  
    val_accuracy = model_history.history['val_accuracy']

    plt.plot(epoch_range, training_loss, '-b', label='training loss')
    plt.plot(epoch_range, training_accuracy, '-r', label='training accuracy')

    plt.plot(epoch_range, val_loss, '--b', label='validation loss')
    plt.plot(epoch_range, val_accuracy, '--r', label='validation accuracy')


    plt.title("Loss and Accuracy by Training Epoch")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(os.path.join(path, f"{plot_title}.png"))
    
    plt.show()


def balance_classes(X_train, y_train): 
    '''
    Handles class imbalance in data by downsampling 
    majority classes

    Returns resampled X_train and y_train sets
    '''
    counts = y_train.value_counts()
    min_counts = counts.min()
    X_trains, y_trains = [],[]
    for cat in range(4):
        y_trains.append(y_train[y_train == cat + 1].sample(n = min_counts, random_state = 42))
        X_trains.append(X_train[y_train == cat + 1].sample(n = min_counts, random_state = 42))
    n = X_trains[0].shape[0]
    X_train = pd.concat(X_trains).sample(n = n, random_state = 42)
    y_train = pd.concat(y_trains).sample(n = n, random_state = 42)
    if not (X_train.index == y_train.index).all():
        raise ValueError('X and y have different indices')
    return X_train, y_train
