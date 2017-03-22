import time

import numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from Data_analisys import  plot_forest_vect, plot_svm_vect
from Preprocessing import TRAIN_DATASET_LABLED, TRAIN_DATASET_UNLABLED, TEST_DATASET
from util import scoring, polish_tfidf_kbest

TO_PLOT=False

def SVC_classifier(train_set_labled,train_set_unlabled,test_set):
    # tratto i dati
    xtrain_vec, xtest_vec, ytrain = polish_tfidf_kbest(train_set_labled, train_set_unlabled, test_set)

    print("inizio classificazione......")
    start = time.time()

    #inizzializzo classificatore
    svc=LinearSVC(verbose= True, penalty="l2",loss="hinge")
    svc=svc.fit(xtrain_vec, ytrain)
    end = time.time()
    print("fitting avvenuto\ntempo impiegato: " + str(end - start))

    if (TO_PLOT):
        plot_svm_vect(svc)

    to_save=svc.coef_
    numpy.savetxt("svm_coeff.txt",to_save)


    pred_forest = svc.predict(xtest_vec)

    scoring(pred_forest)

def multiple_classifier(*models):
    xtrain_vec, xtest_vec, ytrain = polish_tfidf_kbest(TRAIN_DATASET_LABLED, TRAIN_DATASET_UNLABLED, TEST_DATASET)

    for model in models:
        print("Inizio fittaggio per "+str(model))
        model.fit(xtrain_vec, ytrain)
        pred=model.predict(xtest_vec)
        scoring(pred)


def forest_classifier(train_set_labled,train_set_unlabled,test_set):

    #tratto i dati
    xtrain_vec, xtest_vec,ytrain=polish_tfidf_kbest(train_set_labled,train_set_unlabled,test_set)

    print("inizio classificazione......")
    start = time.time()


    # inizzializzo il classificatore e inizio il fittaggio
    forest = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=1, criterion="entropy")
    forest = forest.fit(xtrain_vec, ytrain)
    end = time.time()
    print("fitting avvenuto\ntempo impiegato: " + str(int(end - start)))

    if (TO_PLOT):
        plot_forest_vect(forest)

    # cross_validation_score(forest,xtest_vec)
    # return

    # oob_error = 1 - forest.oob_score_
    # print("oob_error: "+str(oob_error))

    # adesso posso provare a fare la predizione
    pred_forest = forest.predict(xtest_vec)

    scoring(pred_forest)

