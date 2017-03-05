import os
import numpy as np
import hickle as hkl


from Preprocessing import TEST_DATASET


def scoring(prediction):
    return np.mean(prediction == TEST_DATASET["sentiment"])

def polish_tfidf_kbest(train_set_labled,train_set_unlabled,test_set):
    # faccio le divisioni
    xtrainL = train_set_labled["review"]
    xtrainU = train_set_unlabled["review"]
    xtest = test_set["review"]
    ytrain = train_set_labled["sentiment"]



    print("inizio trasformazione da stringa  a vettore......")

    # trasformo da stringhe a vettori
    xtrain_vec, xtest_vec, junk = string2vecTFIDF(xtrainL, xtrainU, xtest)

    print("inizio dimensionality rediction......")

    # eseguo una ricerca delle labels migliori
    xtrain_vec, xtest_vec = dimensionality_reductionKB(xtrain_vec, ytrain, xtest_vec, percentage=0.9)
    return  xtrain_vec, xtest_vec, ytrain



