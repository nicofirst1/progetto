import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

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

def svm_grid(xtrain,ytrain):
    svm=LinearSVC(verbose=1)

    param={"C":[10,100]}
    #"C":[1,10,100],,"probability":[True,False],"shrinking":[True,False],
     #      "decision_function_shape":["ovo","ovr",None]

    grid=GridSearchCV(svm,param,n_jobs=-1,verbose=1,error_score=-1)
    grid.fit(xtrain,ytrain)

    print(grid.best_estimator_)
    print(grid.best_score_)
    print(grid.best_params_)
