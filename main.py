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
def remove_headers_footers(data):
    new_data = []
    for email in data:
        email = email.split(' ')
        email = email[22:-1]
        email = ''.join(email)
        new_data.append(email)
    return new_data

#easy_ham = remove_headers_footers(easy_ham)
#hard_ham = remove_headers_footers(hard_ham)
#spam = remove_headers_footers(spam)

ham = easy_ham + hard_ham

# split the data into train and test
import random as rd
def split_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

X = ham + spam
y = ['ham'] * len(ham) + ['spam'] * len(spam)

hamtrain, hamtest, spamtrain, spamtest = split_data(X, y)

# transform the data with CountVectorizer
def vectorize_data(X_train, X_test):
    vectorizer = CountVectorizer( min_df=2, max_df=0.8)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

# function that trains a Naive Bayes classifier and reports True/False Positives and Negatives rate by
# first vectorizing the data
def train_and_evaluate(X_train, X_test, y_train, y_test):
    X_train, X_test = vectorize_data(X_train, X_test)
    classifier = MultinomialNB(fit_prior=False)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print('Accuracy with Multinomial: ', accuracy_score(y_test, y_pred))
    return y_pred

def vectorise_data_occurance(X_train, X_test):
    vectorizer = CountVectorizer(binary=True, min_df=2)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test

def train_and_evaluate_occurance(X_train, X_test, y_train, y_test):
    X_train, X_test = vectorise_data_occurance(X_train, X_test)
    classifier = BernoulliNB(fit_prior=False)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print('Accuracy with Bernoulli: ', accuracy_score(y_test, y_pred))
    return y_pred

y_predSpamMB = train_and_evaluate(hamtrain, hamtest, spamtrain, spamtest)
y_predSpamB = train_and_evaluate_occurance(hamtrain, hamtest, spamtrain, spamtest)
#y_predHam = train_and_evaluate(spamtrain, spamtest, hamtrain, hamtest)


def true_false_rates(y_test, y_pred):
    TP = np.sum((np.array(y_test) == 'spam') & (y_pred == 'spam'))
    TN = np.sum((np.array(y_test) == 'ham') & (y_pred == 'ham'))
    FP = np.sum((np.array(y_test) == 'ham') & (y_pred == 'spam'))
    FN = np.sum((np.array(y_test) == 'spam') & (y_pred == 'ham'))
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR, TP, TN, FP, FN

spamTPR, spamFPR, spamTP, spamTN, spamFP, spamFN = true_false_rates(spamtest, y_predSpamMB)


print('Multinomial classifier')
print('True Positive Rate: ', spamTPR)
print('False Positive Rate: ', spamFPR)
print('True Positives: ', spamTP)
print('True Negatives: ', spamTN)
print('False Positives: ', spamFP)
print('False Negatives: ', spamFN)

occSpamTPR, occSpamFPR, occSpamTP, occSpamTN, occSpamFP, occSpamFN = true_false_rates(spamtest, y_predSpamB)
print('Bernoulli classifier')
print('True Positive Rate: ', occSpamTPR)
print('False Positive Rate: ', occSpamFPR)
print('True Positives: ', occSpamTP)
print('True Negatives: ', occSpamTN)
print('False Positives: ', occSpamFP)
print('False Negatives: ', occSpamFN)

#spam vs easy ham

Xeasy = easy_ham + spam
yeasy = ['easy_ham'] * len(easy_ham) + ['spam'] * len(spam)

print(len(Xeasy))

easyhamtrain, easyhamtest, spamtrain, spamtest = split_data(Xeasy, yeasy)

y_predEasyHamMB = train_and_evaluate(easyhamtrain, easyhamtest, spamtrain, spamtest)
y_predEasyHamB = train_and_evaluate_occurance(easyhamtrain, easyhamtest, spamtrain, spamtest)

easyhamTPR, easyhamFPR, easyhamTP, easyhamTN, easyhamFP, easyhamFN = true_false_rates(spamtest, y_predEasyHamMB)
print('Multinomial classifier')
print('True Positive Rate: ', easyhamTPR)
print('False Positive Rate: ', easyhamFPR)
print('True Positives: ', easyhamTP)
print('True Negatives: ', easyhamTN)
print('False Positives: ', easyhamFP)
print('False Negatives: ', easyhamFN)

occEasyhamTPR, occEasyhamFPR, occEasyhamTP, occEasyhamTN, occEasyhamFP, occEasyhamFN = true_false_rates(spamtest, y_predEasyHamB)
print('Bernoulli classifier')
print('True Positive Rate: ', occEasyhamTPR)
print('False Positive Rate: ', occEasyhamFPR)
print('True Positives: ', occEasyhamTP)
print('True Negatives: ', occEasyhamTN)
print('False Positives: ', occEasyhamFP)
print('False Negatives: ', occEasyhamFN)

#spam vs hard ham

Xhard = hard_ham + spam
print(len(Xhard))
print(len(hard_ham))
print(len(spam))
yhard = ['hard_ham'] * len(hard_ham) + ['spam'] * len(spam)

hardhamtrain, hardhamtest, spamtrain, spamtest = split_data(Xhard, yhard)

y_predHardHamMB = train_and_evaluate(hardhamtrain, hardhamtest, spamtrain, spamtest)
y_predHardHamB = train_and_evaluate_occurance(hardhamtrain, hardhamtest, spamtrain, spamtest)


hardhamTPR, hardhamFPR, hardhamTP, hardhamTN, hardhamFP, hardhamFN = true_false_rates(spamtest, y_predHardHamMB)
print('Multinomial classifier')
print('True Positive Rate: ', hardhamTPR)
print('False Positive Rate: ', hardhamFPR)
print('True Positives: ', hardhamTP)
print('True Negatives: ', hardhamTN)
print('False Positives: ', hardhamFP)
print('False Negatives: ', hardhamFN)

occHardhamTPR, occHardhamFPR, occHardhamTP, occHardhamTN, occHardhamFP, occHardhamFN = true_false_rates(spamtest, y_predHardHamB)
print('Bernoulli classifier')
print('True Positive Rate: ', occHardhamTPR)
print('False Positive Rate: ', occHardhamFPR)
print('True Positives: ', occHardhamTP)
print('True Negatives: ', occHardhamTN)
print('False Positives: ', occHardhamFP)
print('False Negatives: ', occHardhamFN)

























