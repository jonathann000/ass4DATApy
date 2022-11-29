import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


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


# function to read the files into a list
def read_files_to_list(file_paths):
    data = []
    for path in file_paths:
        with open(path, 'r', encoding='latin-1') as f:
            data.append(f.read())
    return data


# read the files into a list
easy_ham = read_files_to_list(easy_ham_paths)
hard_ham = read_files_to_list(hard_ham_paths)
spam = read_files_to_list(spam_paths)


# remove headers and footers function
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
def split_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# X is features and y is labels
X = ham + spam
y = ['ham'] * len(ham) + ['spam'] * len(spam)

hamtrain, hamtest, spamtrain, spamtest = split_data(X, y)


# transform the data with CountVectorizer
def vectorize_data(X_train, X_test):
    vectorizer = CountVectorizer(min_df=2, max_df=0.8)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test


# function that trains a Naive Bayes classifier and reports True/False Positives and Negatives rate by
# first vectorizing the data
def train_and_evaluate(X_train, X_test, y_train, y_test):
    X_train, X_test = vectorize_data(X_train, X_test)
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print('Accuracy with Multinomial: ', accuracy_score(y_test, y_pred))
    return y_pred


def vectorise_data_bernoulli(X_train, X_test):
    vectorizer = CountVectorizer(binary=True, min_df=2, max_df=0.8)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test


def train_and_evaluate_bernoulli(X_train, X_test, y_train, y_test):
    X_train, X_test = vectorise_data_bernoulli(X_train, X_test)
    classifier = BernoulliNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print('Accuracy with Bernoulli: ', accuracy_score(y_test, y_pred))
    return y_pred

print('Testing on the whole ham dataset:')
print('-------------------')
y_predSpamMB = train_and_evaluate(hamtrain, hamtest, spamtrain, spamtest)
y_predSpamB = train_and_evaluate_bernoulli(hamtrain, hamtest, spamtrain, spamtest)


def true_false_rates(y_test, y_pred):
    TP = np.sum((np.array(y_test) == 'spam') & (y_pred == 'spam'))
    print('True Positives: ', TP)
    TN = np.sum((np.array(y_test) == 'ham') & (y_pred == 'ham'))
    print('True Negatives: ', TN)
    FP = np.sum((np.array(y_test) == 'ham') & (y_pred == 'spam'))
    print('False Positives: ', FP)
    FN = np.sum((np.array(y_test) == 'spam') & (y_pred == 'ham'))
    print('False Negatives: ', FN)
    TPR = TP / (TP + FN)
    print('True Positive Rate: ', TPR)
    FPR = FP / (FP + TN)
    print('False Positive Rate: ', FPR)
    return



print('Multinomial classifier')
true_false_rates(spamtest, y_predSpamMB)
print('-------------------')
print('Bernoulli classifier')
true_false_rates(spamtest, y_predSpamB)
print('-------------------')

# spam vs easy ham
# X is features and y is labels
Xeasy = easy_ham + spam
yeasy = ['ham'] * len(easy_ham) + ['spam'] * len(spam)

easyhamtrain, easyhamtest, spamtrain, spamtest = split_data(Xeasy, yeasy)

print('Testing on the easy ham dataset:')
print('-------------------')
y_predEasyHamMB = train_and_evaluate(easyhamtrain, easyhamtest, spamtrain, spamtest)
y_predEasyHamB = train_and_evaluate_bernoulli(easyhamtrain, easyhamtest, spamtrain, spamtest)


print('Multinomial classifier')
true_false_rates(spamtest, y_predEasyHamMB)
print('-------------------')
print('Bernoulli classifier')
true_false_rates(spamtest, y_predEasyHamB)
print('-------------------')
# spam vs hard ham

# X is features and y is labels
Xhard = hard_ham + spam
yhard = ['ham'] * len(hard_ham) + ['spam'] * len(spam)

hardhamtrain, hardhamtest, spamtrain, spamtest = split_data(Xhard, yhard)

print('Testing on the hard ham dataset:')
print('-------------------')
y_predHardHamMB = train_and_evaluate(hardhamtrain, hardhamtest, spamtrain, spamtest)
y_predHardHamB = train_and_evaluate_bernoulli(hardhamtrain, hardhamtest, spamtrain, spamtest)

print('Multinomial classifier')
true_false_rates(spamtest, y_predHardHamMB)
print('-------------------')
print('Bernoulli classifier')
true_false_rates(spamtest, y_predHardHamB)
print('-------------------')