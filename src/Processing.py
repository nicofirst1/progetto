import time
from os import system

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from src.Data_analisys import plot_svm_dataset, plot_top_forest
from src.Preprocessing import TRAIN_DATASET_LABLED, TRAIN_DATASET_UNLABLED, TEST_DATASET
from src.util import scoring, polish_tfidf_kbest, save_to_csv

# Todo: linear svm vs l'altro
# Todo: analisi dei casi in cui il modello fallisce

TO_PLOT = False
TO_CSV=False



def SVC_classifier(train_set_labled, train_set_unlabled, test_set):
    # tratto i dati
    xtrain_vec, xtest_vec, ytrain, names = polish_tfidf_kbest(train_set_labled, train_set_unlabled, test_set)

    print("inizio classificazione......")
    start = time.time()

    # inizzializzo classificatore e fitto i dati
    svc = LinearSVC(verbose=True, penalty="l2", loss="hinge")
    svc = svc.fit(xtrain_vec, ytrain)

    end = time.time()
    tot = end - start
    print("fitting avvenuto\ntempo impiegato: " + str(int(tot / 60)) + "' " + str(int(tot % 60)) + "''\n")
    system('say "Fittaggio di esse vu emme finito"')

    if TO_PLOT:
        plot_svm_dataset(xtrain_vec, ytrain, svc)
        # plot_svm_vect(svc)



    # eseguo la predizione sul test set
    pred_forest = svc.predict(xtest_vec)
    scoring(pred_forest,TEST_DATASET["sentiment"],"Test set")


    if (TO_CSV):
        save_to_csv(pred_forest)

    # salvo le recensioni incorrettamente predette
    #save_wrong_answer(pred_forest,names,xtest_vec)


def multiple_classifier(*models):
    xtrain_vec, xtest_vec, ytrain, names = polish_tfidf_kbest(TRAIN_DATASET_LABLED,
                                                                   TRAIN_DATASET_UNLABLED, TEST_DATASET)

    for model in models:
        print("Inizio fittaggio per " + str(model))
        model.fit(xtrain_vec, ytrain)
        pred = model.predict(xtest_vec)
        scoring(pred,TEST_DATASET["sentiment"])


def forest_classifier(train_set_labled, train_set_unlabled, test_set):
    # tratto i dati
    xtrain_vec, xtest_vec, ytrain, names = polish_tfidf_kbest(train_set_labled, train_set_unlabled, test_set)

    print("inizio classificazione......")
    start = time.time()

    # inizzializzo il classificatore e inizio il fittaggio
    forest = RandomForestClassifier(n_estimators=250, n_jobs=-1, verbose=1, criterion="entropy")
    forest = forest.fit(xtrain_vec, ytrain)

    end = time.time()
    tot = end - start
    print("fitting avvenuto\ntempo impiegato: " + str(int(tot / 60)) + "' " + str(int(tot % 60)) + "''\n")
    system('say "Fittaggio della foresta randomica completato"')

    if TO_PLOT:
        # plot_forest_vect(forest)
        # plot_trees(forest.estimators_,names)
        plot_top_forest(forest, names, 20)

    # adesso posso provare a fare la predizione per il validatione  test set
    pred_forest = forest.predict(xtest_vec)

    if(TO_CSV):
        pd.DataFrame(data=pred_forest).to_csv(path_or_buf="/Users/nicolo/PycharmProjects/progetto/")


    #save_wrong_answer(pred_forest,names,xtest_vec)

    scoring(pred_forest,TEST_DATASET["sentiment"],"Test Set")
