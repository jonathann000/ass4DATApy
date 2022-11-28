import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# make DATA human readable
import pandas as pd

#preprocessing
from sklearn.preprocessing import LabelEncoder

# split the data into train and test
from sklearn.model_selection import train_test_split

# transform the data with CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import os

from tables.file import File

# function to put file paths into a list
def file_list(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths


# put file paths for the different directories into lists
easy_ham_paths = file_list('DATAhamspam/easy_ham')
hard_ham_paths = file_list('DATAhamspam/hard_ham')
spam_paths = file_list('DATAhamspam/spam')


# read the files into a list
def read_files_to_list(file_paths):
    data = []
    for path in file_paths:
        with open(path, 'r', encoding= 'latin-1') as f:
            data.append(f.read())
    return data

# read the files into a list
easy_ham = read_files_to_list(easy_ham_paths)
hard_ham = read_files_to_list(hard_ham_paths)
spam = read_files_to_list(spam_paths)

# allting fungerar fram tills hit


#split the data into features and labels with CountVectorizer?


# split the data into train and test
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


hamtrain, hamtest, spamtrain, spamtest = split_data(easy_ham)

# transform the data with CountVectorizer
def vectorize_data(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

# function that trains a Naive Bayes classifier and reports True/False Positives and Negatives rate by
# first vectorizing the data
def train_and_evaluate(X_train, X_test, y_train, y_test):
    X_train, X_test = vectorize_data(X_train, X_test)
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    return y_pred








