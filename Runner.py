import numpy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.svm import LinearSVC
from Data_analisys import plot_learning_curve, plot_chi2_top, plot_top_n_words, plot_svm_dataset
from Preprocessing import TRAIN_DATASET_LABLED, TRAIN_DATASET_UNLABLED, sentences_polishing
from Processing import forest_classifier, multiple_classifier, SVC_classifier
from util import *

## TRATTAMENTO DATI
# x_train_vec, x_test_vec, vect=string2vecTFIDF(TRAIN_DATASET_LABLED["review"],TRAIN_DATASET_UNLABLED["review"],TEST_DATASET["review"])
x_train_vec, x_test_vec, ytrain = polish_tfidf_kbest(TRAIN_DATASET_LABLED, TRAIN_DATASET_UNLABLED, TEST_DATASET)

## PLOTS
# plot_chi2_vect(x_train_vec,TRAIN_DATASET_LABLED["sentiment"])
# plot_chi2_top(x_train_vec,TRAIN_DATASET_LABLED["sentiment"],vect)
# plot_top_n_words(vect,10)

## CLASSIFICAZIONE
# forest_classifier(TRAIN_DATASET_LABLED,TRAIN_DATASET_UNLABLED,TEST_DATASET)
# SVC_classifier(TRAIN_DATASET_LABLED,TRAIN_DATASET_UNLABLED,TEST_DATASET)


## PLOTTING LEARNING RATE
forest = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=1, criterion="entropy")
#svc=LinearSVC(verbose= True, penalty="l2",loss="hinge")

plot_learning_curve(forest,"Forest learning curve",x_train_vec,ytrain)
#plot_learning_curve(svc,"SVC learning curve",x_train_vec,ytrain)

# svm_grid(x_test_vec,TRAIN_DATASET_LABLED["sentiment"])

## PROVA
# multiple_classifier(LinearSVC(verbose= True, penalty="l2",loss="hinge"))


from os import system

system('say "Esecuzione terminata"')
system('say "Esecuzione terminata"')
