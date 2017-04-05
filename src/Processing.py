import time
from os import system

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from src.Data_analisys import plot_svm_dataset, plot_top_forest, plot_trees, plot_forest_vect
from src.Preprocessing import TRAIN_DATASET_LABLED, TRAIN_DATASET_UNLABLED, TEST_DATASET
from src.util import scoring, polish_tfidf_kbest, save_to_csv, save_wrong_answer

# Todo: linear svm vs l'altro
# Todo: analisi dei casi in cui il modello fallisce

TO_PLOT = False
TO_CSV=False
TO_SAVE_WRONG=False


def SVC_classifier(train_set_labled, train_set_unlabled, test_set):
    # Data pre processing
    xtrain_vec, xtest_vec, ytrain, names = polish_tfidf_kbest(train_set_labled, train_set_unlabled, test_set)

    print("Executing classification......")
    start = time.time()

    # classifier initialization and fitting
    svc = LinearSVC(verbose=True, penalty="l2", loss="hinge")
    svc = svc.fit(xtrain_vec, ytrain)

    end = time.time()
    tot = end - start
    print("fitting completed\nTotal time: " + str(int(tot / 60)) + "' " + str(int(tot % 60)) + "''\n")
    system('say "Fitting for SMV completed"')

    if TO_PLOT:
        plot_svm_dataset(xtrain_vec, ytrain, svc)
        # plot_svm_vect(svc)



    # prediction
    pred_forest = svc.predict(xtest_vec)
    scoring(pred_forest,TEST_DATASET["sentiment"],"Test set")


    if (TO_CSV):
        save_to_csv(pred_forest)
    if(TO_SAVE_WRONG):
     #saving wrong prediction for analysis
        save_wrong_answer(pred_forest,names,xtest_vec)


def multiple_classifier(*models):
    xtrain_vec, xtest_vec, ytrain, names = polish_tfidf_kbest(TRAIN_DATASET_LABLED,
                                                                   TRAIN_DATASET_UNLABLED, TEST_DATASET)

    for model in models:
        print("Executing fitting dor " + str(model))
        model.fit(xtrain_vec, ytrain)
        pred = model.predict(xtest_vec)
        scoring(pred,TEST_DATASET["sentiment"])


def forest_classifier(train_set_labled, train_set_unlabled, test_set):
    # Data pre processing
    xtrain_vec, xtest_vec, ytrain, names = polish_tfidf_kbest(train_set_labled, train_set_unlabled, test_set)

    print("Executing classification......")
    start = time.time()

    # classifier initialization and fitting
    forest = RandomForestClassifier(n_estimators=250, n_jobs=-1, verbose=1, criterion="entropy")
    forest = forest.fit(xtrain_vec, ytrain)

    end = time.time()
    tot = end - start
    print("fitting completed\nTotal time: " + str(int(tot / 60)) + "' " + str(int(tot % 60)) + "''\n")
    system('say "Fitting for random forest completed"')

    if TO_PLOT:
        plot_forest_vect(forest)
        plot_trees(forest.estimators_,names)
        plot_top_forest(forest, names, 20)

    # prediction
    pred_forest = forest.predict(xtest_vec)

    if(TO_CSV):
        pd.DataFrame(data=pred_forest).to_csv(path_or_buf="/Users/nicolo/PycharmProjects/progetto/")

    if TO_SAVE_WRONG:
        save_wrong_answer(pred_forest,names,xtest_vec)

    scoring(pred_forest,TEST_DATASET["sentiment"],"Test Set")
