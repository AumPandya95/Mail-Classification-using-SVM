
# coding: utf-8

import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

os.chdir('D://PGDBA/Assignments_Projects/SVM_MailClassification/')

def make_dictionary(train_dir):
    #Make a list to store all the words
    all_words=[]
    #Extract file names from the training folder
    emails=[os.path.join(train_dir,f) for f in os.listdir(train_dir)]
    for mail in emails:
        with open(mail) as m:
            for line in m:
                words = line.split()
                all_words += words        #Append all the words into the list
    dictionary = Counter(all_words)       #collections.Counter() counts the frequency of words
    list_to_remove = list(dictionary)     #Converts the dict() to a list()

    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]          #del if the item is a number
        elif len(item) ==1:
            del dictionary[item]          #del if the word has a length of 1
        
    dictionary = dictionary.most_common(3000)      #Output is a tuple 
   
    return dictionary

def extract_features(mail_dir):
    #Extract file names from the training folder
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    #Create a matrix with dimensions(no.of files X no.of words)
    features_matrix = np.zeros((len(files),3000))
    #Classification variable
    train_labels = np.zeros(len(files))
    count = 0
    docID = 0
    for fil in files:
        with open(fil) as fi:
            for i,line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i,d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID,wordID] = words.count(word)
        train_labels[docID] = 0;
        filepathTokens = fil.split('\\')
        lastToken = filepathTokens[len(filepathTokens) - 1]
        if lastToken.startswith("spmsg"):
            train_labels[docID] = 1;
            count = count + 1
        docID = docID + 1
    return features_matrix, train_labels

Train_directory = 'train-mails'
Test_directory = 'test-mails'

dictionary = make_dictionary(Train_directory)
print("Reading and processing mails from the file.",'\n')

features_matrix,labels = extract_features(Train_directory)
test_feature_matrix,test_labels = extract_features(Test_directory)

model = svm.SVC(kernel = 'linear')
print("Training model...",'\n')
model.fit(features_matrix,labels)

predicted_labels = model.predict(test_feature_matrix)
print("Finished classifying.")
print("The accuracy score is: ",accuracy_score(test_labels,predicted_labels))

