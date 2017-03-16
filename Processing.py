import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import *
from sklearn.svm import SVC

from Data_analisys import plot_SGD_vect, plot_forest_vect, plot_SGD_decision
from Preprocessing import string2vecCV
from util import scoring, polish_tfidf_kbest, cross_validation_score

TO_PLOT=True


def forest_classifier(train_set_labled,train_set_unlabled,test_set):

    #tratto i dati
    xtrain_vec, xtest_vec,ytrain=polish_tfidf_kbest(train_set_labled,train_set_unlabled,test_set)

    print("inizio classificazione......")
    start = time.time()


    # inizzializzo il classificatore e inizio il fittaggio
    forest = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=1, criterion="entropy")
    forest = forest.fit(xtrain_vec, ytrain)
    end = time.time()
    print("fitting avvenuto\ntempo impiegato: " + str(end - start))

    if (TO_PLOT):
        plot_forest_vect(forest)

    # cross_validation_score(forest,xtest_vec)
    # return

    # oob_error = 1 - forest.oob_score_
    # print("oob_error: "+str(oob_error))

    # adesso posso provare a fare la predizione
    pred_forest = forest.predict(xtest_vec)

    scoring(pred_forest)

