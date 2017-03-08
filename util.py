import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from Preprocessing import TEST_DATASET, string2vecTFIDF, dimensionality_reductionKB

def cross_validation_score(model, x):
    scores=cross_val_score(model,x,y=TEST_DATASET["sentiment"])
    print(scores)


def scoring(prediction):
    pred=str(np.mean(prediction == TEST_DATASET["sentiment"]))
    print("il risultato della predizione è: "+pred)

def polish_tfidf_kbest(train_set_labled,train_set_unlabled,test_set):
    # faccio le divisioni
    xtrainL = train_set_labled["review"]
    xtrainU = train_set_unlabled["review"]
    xtest = test_set["review"]
    ytrain = train_set_labled["sentiment"]



    print("inizio trasformazione da stringa  a vettore......")

    # trasformo da stringhe a vettori
    xtrain_vec, xtest_vec, vect = string2vecTFIDF(xtrainL, xtrainU, xtest)

    print("inizio dimensionality reduction......")

    # eseguo una ricerca delle labels migliori
    xtrain_vec, xtest_vec = dimensionality_reductionKB(xtrain_vec, ytrain, xtest_vec, percentage=0.9)
    return  xtrain_vec, xtest_vec, ytrain


def grid_search_SGD(xtrain,ytrain):

    sgd=SGDClassifier(verbose=1,n_jobs=-1,loss="modified_huber",random_state=4,n_iter=10,
                      shuffle=True)
    param={"alpha":[0.0001,0.0002,0.00009]
           }
    grid=GridSearchCV(sgd,param,error_score=0,verbose=True)
    grid.fit(xtrain,ytrain)
    print(grid.best_params_)
    print(grid.best_score_)
