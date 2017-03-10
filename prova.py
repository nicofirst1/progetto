from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit

from Data_analisys import plot_learning_curve
from Preprocessing import TRAIN_DATASET_LABLED, TRAIN_DATASET_UNLABLED
from Processing import SGD_classifier, forest_classifier
from util import *

## TRATTAMENTO DATI
x_train_vec, x_test_vec, vect=string2vecTFIDF(TRAIN_DATASET_LABLED["review"],TRAIN_DATASET_UNLABLED["review"],TEST_DATASET["review"])
#
# plot_chi2_vect(x_train_vec,TRAIN_DATASET_LABLED["sentiment"])
#

## CLASSIFICAZIONE
#forest_classifier(TRAIN_DATASET_LABLED,TRAIN_DATASET_UNLABLED,TEST_DATASET)
# SGD_classifier(TRAIN_DATASET_LABLED,TRAIN_DATASET_UNLABLED,TEST_DATASET)
# #naive_bayes(TRAIN_DATASET_LABLED,TEST_DATASET)

## PLOT SGD
# plot_SGD_top(x_train_vec,TRAIN_DATASET_LABLED["sentiment"],x_test_vec,vect)

# TUNING SGD
# xtrain_vec, xtest_vec, ytrain=polish_tfidf_kbest(TRAIN_DATASET_LABLED,TRAIN_DATASET_UNLABLED,TEST_DATASET)
# grid_search_SGD(xtrain_vec,ytrain)

## PLOTTING LEARNING RATE
sgd=SGDClassifier(verbose=1,n_jobs=-1,loss="modified_huber",random_state=4,n_iter=10,shuffle=True)
#forest = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=1, criterion="entropy")

#plot_learning_curve(forest,"Forest learning curve",x_train_vec,TRAIN_DATASET_LABLED["sentiment"])
plot_learning_curve(sgd,"SGD learning curve",x_train_vec,TRAIN_DATASET_LABLED["sentiment"])
